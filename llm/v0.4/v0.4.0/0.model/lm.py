"""v0.4.0: 바이트 BPE + BOS/EOS 문서 경계 + 확률 하한 없는 평가 계약."""
import importlib.util
import json
import math
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


previous = module('v040_previous', ROOT/'llm/v0.3/v0.3.4/0.model/lm.py')
tokenizer = module('v040_tokenizer', HERE/'tokenizer.py')
ByteBPE = tokenizer.ByteBPE
sequence_loss = previous.sequence_loss
_data = previous._data


class BPENetwork(previous.TransformerLM):
    def forward(self, tokens):
        logits = super().forward(tokens)
        ids = torch.arange(logits.shape[-1], device=logits.device)
        # 역할 토큰은 예약만 합니다. 현재 산문 모델의 생성 후보는 바이트/BPE와 EOS.
        return logits.masked_fill((ids < len(tokenizer.SPECIALS)) & ids.ne(1), float('-inf'))


class NeuralLM(previous.NeuralLM):
    MODEL_VERSION = 'v0.4.0'
    EVALUATION_CONTRACT = 'byte-bpe-bos-eos-raw-nll-v1'
    PAD, END, BOS, USER, BOT, SYSTEM = tokenizer.SPECIALS
    VOCAB_SIZE = 768
    EPOCHS = 30
    PATIENCE = 5
    BATCH_SIZE = 128

    def __init__(self):
        super().__init__()
        self.bpe = None

    def tokenizer_name(self):
        return 'byte-bpe-v1'

    def read_sentences(self, path):
        return self.documents_from_bytes(Path(path).read_bytes())

    @staticmethod
    def documents_from_bytes(source):
        # LF는 문서 레코드 구분자입니다. CR·공백·빈 문서는 원문 그대로 보존합니다.
        raw = source.decode('utf-8')
        if not raw:
            return []
        records = raw.split('\n')
        return records[:-1] if raw.endswith('\n') else records

    def train(self, sentences, valid_sentences=None):
        sentences = list(sentences)
        self.bpe = ByteBPE.train(sentences, self.VOCAB_SIZE)
        return super().train(sentences, valid_sentences)

    def build_vocab(self, sentences):
        if self.bpe is None:
            self.bpe = ByteBPE.train(sentences, self.VOCAB_SIZE)
        return self.bpe.tokens.copy()

    def tokenize(self, text):
        if self.bpe is None:
            raise ValueError('먼저 토크나이저를 학습하거나 모델을 로드하세요.')
        return [self.bpe.tokens[i] for i in self.bpe.encode(text)]

    def detokenize(self, tokens):
        return self.bpe.decode([self.stoi[t] for t in tokens], skip_special=True)

    def prepare(self, tokens):
        # 원문 속 '<EOS>' 등은 encode가 일반 바이트로 취급하므로 충돌하지 않습니다.
        return [self.BOS] + list(tokens) + [self.END]

    def is_end(self, token):
        return token == self.END

    def build_net(self):
        return BPENetwork(len(self.itos), self.EMBED, self.BLOCK_SIZE, self.HEADS,
                          self.FFN_HIDDEN, self.DROPOUT, self.LAYERS)

    def extra_metadata(self):
        return {**super().extra_metadata(), 'bpe': self.bpe.to_dict(),
                'evaluation_contract': self.EVALUATION_CONTRACT}

    def restore_extra_metadata(self, meta):
        super().restore_extra_metadata(meta)
        if meta.get('evaluation_contract') != self.EVALUATION_CONTRACT:
            raise ValueError('평가 계약 불일치')
        self.bpe = ByteBPE.from_dict(meta['bpe'])
        self.VOCAB_SIZE = self.bpe.requested_vocab_size
        if self.itos != self.bpe.tokens:
            raise ValueError('체크포인트 어휘와 BPE가 다릅니다.')

    def save(self, model_path, vocab_path=None):
        super().save(model_path, vocab_path)
        path = Path(model_path).parent
        self.bpe.save(path/'tokenizer.json')
        config = {'version': self.MODEL_VERSION, 'evaluation_contract': self.EVALUATION_CONTRACT,
                  'block_size': self.BLOCK_SIZE, 'embed': self.EMBED,
                  **self.extra_metadata()}
        (path/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    def evaluate_rows(self, rows, top_k=5):
        if top_k < 1:
            raise ValueError('top_k는 양수여야 합니다.')
        if self.net is None:
            raise ValueError('모델을 먼저 학습하거나 로드하세요.')
        n = hits = hitsk = 0
        total_nll = text_nll = eos_nll = 0.0
        was_training = self.net.training
        self.net.eval()
        try:
            with torch.no_grad():
                for x,y,known in self.batches(rows,self.EVAL_BATCH):
                    active = y.ne(-100)
                    if not known[active].all():
                        raise ValueError('BPE 평가에서 미등록 토큰이 발견됐습니다.')
                    logits = self.net(x.to(self.device())).cpu()[active].double()
                    targets = y[active]
                    nll = -logits.log_softmax(-1).gather(1,targets[:,None]).squeeze(1)
                    eos = targets.eq(self.stoi[self.END])
                    total_nll += nll.sum().item()
                    text_nll += nll[~eos].sum().item()
                    eos_nll += nll[eos].sum().item()
                    ranked = logits.topk(min(top_k,len(self.itos)-5),dim=-1).indices
                    n += len(targets)
                    hits += int((ranked[:,0]==targets).sum())
                    hitsk += int((ranked==targets[:,None]).any(-1).sum())
        finally:
            self.net.train(was_training)
        mean = total_nll/n if n else None
        ppl = math.exp(mean) if mean is not None and mean<709 else float('inf')
        return {'ppl':ppl,'token_nll':mean,'total_nll':total_nll,'text_nll':text_nll,
                'eos_nll':eos_nll,'n':n,'top1':hits/n if n else 0,
                'topk':hitsk/n if n else 0,'coverage':1.0 if n else 0,'k':top_k,
                'evaluation_contract':self.EVALUATION_CONTRACT}

    def evaluate(self, sentences, top_k=5):
        sentences = list(sentences)
        result = self.evaluate_rows(self.make_windows(sentences),top_k)
        byte_count = sum(len(s.encode('utf-8')) for s in sentences)
        result.update({'text_bytes':byte_count,'documents':len(sentences),
                       'nll_per_byte':result['text_nll']/byte_count if byte_count else None,
                       'bits_per_byte':result['text_nll']/byte_count/math.log(2) if byte_count else None})
        return result

    def token_prob(self, recent, token):
        ids = self._context_ids(recent)
        if ids is None or token not in self.stoi:
            raise ValueError('BPE 토큰/문맥이 유효하지 않습니다.')
        probs = self._probs(recent)
        return probs[self.stoi[token]].item()  # 기존 FLOOR를 적용하지 않습니다.

    def next_dist(self, recent):
        probs = self._probs(recent)
        if probs is None:
            return None
        return {t:probs[i].item() for i,t in enumerate(self.itos) if i==1 or i>=len(tokenizer.SPECIALS)}

    def generate(self, start_text, temperature=0.0, top_k=0, top_p=1.0):
        self._top_k,self._top_p = top_k,top_p
        recent = [self.BOS] + self.tokenize(start_text)
        for _ in range(self.MAX_LENGTH):
            token = self.next_token(recent,temperature)
            if token is None or self.is_end(token):break
            recent.append(token)
        # 무작위 바이트 출력/길이 제한이 UTF-8 문자를 자를 수 있습니다.
        # 입력 round-trip은 strict, 생성 결과 표시만 replacement를 허용합니다.
        return self.bpe.decode([self.stoi[t] for t in recent],skip_special=True,errors='replace')

    def chat(self, user_message, history=None, temperature=0.0, top_k=0, top_p=1.0):
        raise NotImplementedError('v0.4.0은 산문 이어쓰기 모델입니다. 역할 토큰은 예약만 했으며 SFT는 v0.5 범위입니다.')


NGramLM = Model = NeuralLM
DATA_PATH,VALID_PATH = previous.DATA_PATH,previous.VALID_PATH
MODEL_PATH = str(HERE/'model.pt')
VOCAB_PATH = str(HERE/'vocab.json')
