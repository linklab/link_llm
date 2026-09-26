"""AdamW, token-aware logs and exact batch-boundary training resume."""
import importlib.util
import math
import random
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location('v042_previous', HERE.parents[1]/'v0.4.1/0.model/lm.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
module, corpus, packing = previous.module, previous.corpus, previous.packing
ByteBPE, sequence_loss = previous.ByteBPE, previous.sequence_loss


class NeuralLM(previous.NeuralLM):
    MODEL_VERSION = 'v0.4.2'
    WEIGHT_DECAY, GRAD_CLIP, WARMUP_STEPS = .01, 1., 10
    EPOCHS, PATIENCE = 6, 0

    def training_config(self):
        keys = ('EPOCHS', 'BATCH_SIZE', 'LR', 'SEED', 'VOCAB_SIZE', 'EMBED',
                'BLOCK_SIZE', 'HEADS', 'LAYERS', 'FFN_HIDDEN', 'DROPOUT',
                'WEIGHT_DECAY', 'GRAD_CLIP', 'WARMUP_STEPS', 'PATIENCE')
        return {k: getattr(self, k) for k in keys}

    def train(self, sentences, valid_sentences=None, *, resume=None,
              checkpoint=None, max_steps=None):
        train, valid = list(sentences), list(valid_sentences or [])
        config = self.training_config()
        if (not train or self.EPOCHS < 1 or self.BATCH_SIZE < 1 or self.PATIENCE != 0
                or self.WARMUP_STEPS < 0 or self.WEIGHT_DECAY < 0
                or not all(math.isfinite(v) for v in (self.LR, self.GRAD_CLIP, self.WEIGHT_DECAY))
                or self.LR <= 0 or self.GRAD_CLIP <= 0 or (max_steps is not None and max_steps < 1)):
            raise ValueError('invalid training configuration; fixed horizon requires patience=0')
        if {corpus.duplicate_key(t) for t in train} & {corpus.duplicate_key(t) for t in valid}:
            raise ValueError('train/validation duplicate documents')
        identity = {'train': corpus.documents_hash(train), 'valid': corpus.documents_hash(valid),
                    'manifest': self.corpus_manifest_sha256, 'version': self.MODEL_VERSION}
        torch.manual_seed(self.SEED)
        random.seed(self.SEED)
        self.bpe = ByteBPE.train(train, self.VOCAB_SIZE)
        self.initialize(train)
        packs = packing.make_packs(self, train)
        self.packing_stats = packing.statistics(self, train, packs)
        horizon = math.ceil(len(packs)/self.BATCH_SIZE)*self.EPOCHS
        warmup = min(self.WARMUP_STEPS, horizon)

        def rate(step):
            if step < warmup:
                return (step+1)/warmup
            return .1 + .9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,horizon-warmup-1)))

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
        generator = torch.Generator().manual_seed(self.SEED)
        state = {'epoch': 1, 'cursor': 0, 'order': [], 'step': 0, 'loss_sum': 0.,
                 'tokens': 0, 'history': [], 'steps': [], 'best': float('inf'),
                 'best_epoch': 0, 'best_weights': None}
        if resume:
            saved = torch.load(resume, map_location='cpu', weights_only=True)
            if (saved['format'] != 1 or saved['config'] != config or saved['identity'] != identity
                    or saved['tokenizer'] != self.bpe.to_dict()
                    or saved['device'] != str(self.device())
                    or saved['torch'] != str(torch.__version__)
                    or saved['threads'] != torch.get_num_threads()):
                raise ValueError('resume config/data/tokenizer/runtime mismatch')
            self.net.load_state_dict(saved['weights'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            state = saved['state']
            generator.set_state(saved['sampler_rng'])
            torch.set_rng_state(saved['torch_rng'])
            random.setstate(saved['python_rng'])
            if saved['cuda_rng'] is not None:
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
            if saved['mps_rng'] is not None:
                torch.mps.set_rng_state(saved['mps_rng'])
        initial_step = state['step']
        valid_rows = self.make_windows(valid) if valid else None
        while state['epoch'] <= self.EPOCHS:
            if not state['order']:
                state['order'] = torch.randperm(len(packs), generator=generator).tolist()
            self.net.train()
            stop = False
            while state['cursor'] < len(packs):
                indices = state['order'][state['cursor']:state['cursor']+self.BATCH_SIZE]
                batch = {k: v.to(self.device()) for k,v in packing.collate([packs[i] for i in indices]).items()}
                optimizer.zero_grad(set_to_none=True)
                loss = sequence_loss(self.net(batch['x'], batch['segments'], batch['positions']), batch['y'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.GRAD_CLIP, error_if_nonfinite=True)
                lr = optimizer.param_groups[0]['lr']
                optimizer.step()
                scheduler.step()
                count = int(batch['y'].ne(-100).sum())
                state['step'] += 1
                state['cursor'] += len(indices)
                state['tokens'] += count
                state['loss_sum'] += loss.item()*count
                state['steps'].append({'step': state['step'], 'tokens': count, 'loss': loss.item(),
                                       'lr': lr, 'grad_norm_before_clip': float(norm)})
                if max_steps is not None and state['step']-initial_step >= max_steps:
                    stop = True
                    break
            if state['cursor'] == len(packs):
                metrics = self.evaluate_rows(valid_rows) if valid_rows else None
                score = metrics['ppl'] if metrics else state['loss_sum']/state['tokens']
                if not math.isfinite(score):
                    raise FloatingPointError('nonfinite validation score')
                state['history'].append({'epoch': state['epoch'], 'train_loss': state['loss_sum']/state['tokens'],
                                         'valid_ppl': metrics['ppl'] if metrics else None})
                if score < state['best']:
                    state['best'], state['best_epoch'] = score, state['epoch']
                    state['best_weights'] = {k:v.detach().cpu().clone() for k,v in self.net.state_dict().items()}
                print(state['history'][-1], flush=True)
                state.update(epoch=state['epoch']+1, cursor=0, order=[], tokens=0, loss_sum=0.)
            if checkpoint:
                path = Path(checkpoint)
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {'format': 1, 'config': config, 'identity': identity, 'tokenizer': self.bpe.to_dict(),
                           'device': str(self.device()), 'torch': str(torch.__version__), 'threads': torch.get_num_threads(),
                           'weights': self.net.state_dict(), 'optimizer': optimizer.state_dict(),
                           'scheduler': scheduler.state_dict(), 'state': state,
                           'sampler_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
                           'python_rng': random.getstate(),
                           'cuda_rng': torch.cuda.get_rng_state_all() if self.device().type == 'cuda' else None,
                           'mps_rng': torch.mps.get_rng_state() if self.device().type == 'mps' else None}
                temporary = path.with_suffix('.tmp')
                torch.save(payload, temporary)
                temporary.replace(path)
            if stop:
                break
        self.invocation_tokens = sum(s['tokens'] for s in state['steps'][initial_step:])
        self.completed = state['epoch'] > self.EPOCHS
        self.history, self.step_history, self.best_epoch = state['history'], state['steps'], state['best_epoch']
        if self.completed:
            self.net.load_state_dict(state['best_weights'])
        self.net.eval()
        return self.net


Model = NGramLM = NeuralLM
MODEL_PATH, VOCAB_PATH = str(HERE/'model.pt'), str(HERE/'vocab.json')
DATA_PATH, VALID_PATH = previous.DATA_PATH, previous.VALID_PATH
