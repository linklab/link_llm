"""v0.4.1: isolated document packing with the v0.4.0 inference/evaluation contract."""
import importlib.util
import math
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


previous = module('v041_previous', HERE.parents[1]/'v0.4.0/0.model/lm.py')
packing = module('v041_packing', HERE/'packing.py')
corpus = module('v041_corpus', ROOT/'data/pretrain/v0.4.1/corpus.py')
ByteBPE = previous.ByteBPE
tokenizer = previous.tokenizer
sequence_loss = previous.sequence_loss


class PackedNetwork(previous.BPENetwork):
    def forward(self, tokens, segments=None, positions=None):
        if segments is None and positions is None:
            return super().forward(tokens)
        if (segments is None or positions is None or tokens.ndim != 2
                or segments.shape != tokens.shape or positions.shape != tokens.shape
                or not 0 < tokens.shape[1] <= self.block_size):
            raise ValueError('packed tensors must all have shape (B,T), 1 <= T <= block_size')
        valid = tokens.ne(0)
        mask = valid.unsqueeze(-1)
        length = tokens.shape[1]
        causal = torch.ones(length, length, device=tokens.device, dtype=torch.bool).tril()
        allowed = (causal[None, None] & valid[:, None, :, None]
                   & valid[:, None, None, :]
                   & segments[:, None, :, None].eq(segments[:, None, None, :]))
        h = (self.embedding(tokens) + self.position(positions)) * mask
        for block in self.blocks:
            normalized = block.ln1(h)

            def split(layer):
                return layer(normalized).reshape(tokens.shape[0], length, block.heads,
                                                 block.head_dim).transpose(1, 2)

            q, k, v = split(block.query), split(block.key), split(block.value)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(block.head_dim)
            scores = scores.masked_fill(~allowed, float('-inf'))
            scores = scores.masked_fill(~allowed.any(-1, keepdim=True), 0)
            weights = scores.softmax(-1).masked_fill(~allowed, 0)
            merged = (weights @ v).transpose(1, 2).reshape(tokens.shape[0], length, -1)
            h = (h + block.dropout(block.projection(merged) * mask)) * mask
            h = (h + block.dropout(block.ffn(block.ln2(h)))) * mask
        logits = self.head(self.final_norm(h) * mask)
        ids = torch.arange(logits.shape[-1], device=tokens.device)
        return logits.masked_fill((ids < 6) & ids.ne(1), float('-inf'))


class NeuralLM(previous.NeuralLM):
    MODEL_VERSION = 'v0.4.1'
    PACKING_POLICY = packing.POLICY

    def __init__(self):
        super().__init__()
        self.corpus_manifest_sha256 = None
        self.packing_stats = {}

    def build_net(self):
        return PackedNetwork(len(self.itos), self.EMBED, self.BLOCK_SIZE, self.HEADS,
                             self.FFN_HIDDEN, self.DROPOUT, self.LAYERS)

    def read_sentences(self, path):
        if Path(path).suffix == '.jsonl':
            return [r['text'] for r in corpus.read_jsonl(path)]
        return super().read_sentences(path)

    def extra_metadata(self):
        return {**super().extra_metadata(), 'packing_policy': self.PACKING_POLICY,
                'corpus_manifest_sha256': self.corpus_manifest_sha256}

    def restore_extra_metadata(self, meta):
        super().restore_extra_metadata(meta)
        if meta.get('packing_policy') != self.PACKING_POLICY:
            raise ValueError('packing policy mismatch')
        self.corpus_manifest_sha256 = meta.get('corpus_manifest_sha256')

    def train(self, sentences, valid_sentences=None):
        sentences = list(sentences)
        valid = list(valid_sentences) if valid_sentences is not None else None
        if not sentences or (valid is not None and not valid):
            raise ValueError('train and supplied validation must contain documents')
        if self.EPOCHS < 1 or self.BATCH_SIZE < 1 or not math.isfinite(self.LR) or self.LR <= 0:
            raise ValueError('epochs, batch_size and lr must be positive')
        # Also protect direct API calls; exact + NFC/whitespace-equivalent duplicates.
        if valid is not None and ({corpus.duplicate_key(t) for t in sentences}
                                  & {corpus.duplicate_key(t) for t in valid}):
            raise ValueError('train/validation duplicate documents')
        torch.manual_seed(self.SEED)
        self.bpe = ByteBPE.train(sentences, self.VOCAB_SIZE)
        self.initialize(sentences)
        packs = packing.make_packs(self, sentences)
        self.packing_stats = packing.statistics(self, sentences, packs)
        loader = packing.batches(packs, self.BATCH_SIZE, shuffle=True, seed=self.SEED)
        valid_rows = self.make_windows(valid) if valid is not None else None
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.LR)
        best, state, bad = float('inf'), None, 0
        self.history, self.best_epoch = [], 0
        for epoch in range(1, self.EPOCHS + 1):
            self.net.train()
            total, count = 0., 0
            for batch in loader:
                b = {key: value.to(self.device()) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = sequence_loss(self.net(b['x'], b['segments'], b['positions']), b['y'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite training loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in self.net.parameters()):
                    raise FloatingPointError('nonfinite gradient')
                optimizer.step()
                n = int(b['y'].ne(-100).sum())
                total += loss.item()*n
                count += n
            metrics = self.evaluate_rows(valid_rows) if valid_rows is not None else None
            score = metrics['ppl'] if metrics else total/count
            if not math.isfinite(score):
                raise FloatingPointError('nonfinite model selection score')
            self.history.append({'epoch': epoch, 'train_loss': total/count,
                                 'valid_ppl': metrics['ppl'] if metrics else None})
            if score < best:
                best, bad, self.best_epoch = score, 0, epoch
                state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
            print(f'epoch {epoch:3d} packed_loss={total/count:.4f} selection={score:.4f}', flush=True)
            if metrics and self.PATIENCE > 0 and bad >= self.PATIENCE:
                break
        self.net.load_state_dict(state)
        self.net.eval()
        return self.net


Model = NGramLM = NeuralLM
MODEL_PATH = str(HERE/'model.pt')
VOCAB_PATH = str(HERE/'vocab.json')
DATA_PATH = str(ROOT/'data/pretrain/v0.4.1/train.jsonl')
VALID_PATH = str(ROOT/'data/pretrain/v0.4.1/valid.jsonl')
