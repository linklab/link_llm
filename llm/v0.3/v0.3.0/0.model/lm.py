"""v0.3.0 단일 헤드 어텐션 LM. 외부 인터페이스와 시퀀스 엔진의 연결부.

카운트 계보의 토크나이저·END·chat만 재사용합니다. MLP의 (B,V) 평가나
학습 코드를 상속하지 않고 (B,T,V) 계약을 명시적으로 구현합니다.
"""
import importlib.util
import json
import math
from pathlib import Path

try:
    import torch
except ImportError as exc:
    raise SystemExit("v0.3.0은 PyTorch가 필요합니다. torch가 설치된 Python을 사용하세요.") from exc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _module("v030_text", ROOT / "llm/v0.0/v0.0.9/0.model/lm.py")
_network = _module("v030_network", HERE / "network.py")
_data = _module("v030_sequences", HERE / "sequences.py")
_training = _module("v030_training", HERE / "training.py")
SingleHeadLM = _network.SingleHeadLM
sequence_loss = _network.sequence_loss


class NeuralLM(_base.NGramLM):
    MODEL_VERSION = "v0.3.0"
    PAD = "<PAD>"
    BLOCK_SIZE = 32
    EMBED = 64
    EPOCHS = 60
    BATCH_SIZE = 64
    EVAL_BATCH = 64
    LR = 0.003
    PATIENCE = 10
    SEED = 1234
    DEVICE = "auto"
    MAX_LENGTH = 40
    sequence_loss = staticmethod(sequence_loss)
    batches = staticmethod(_data.batches)

    def __init__(self):
        super().__init__()
        self.net = None
        self.itos, self.stoi = [], {}
        self.history = []
        self.best_epoch = 0

    def device(self):
        if self.net is not None:
            return next(self.net.parameters()).device
        if self.DEVICE != "auto":
            return torch.device(self.DEVICE)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def build_vocab(self, sentences):
        vocab = [self.PAD]
        seen = {self.PAD}
        for sentence in sentences:
            tokens = self.prepare(self.tokenize(sentence))
            if self.PAD in tokens:
                raise ValueError("<PAD>는 배치용 예약 토큰입니다.")
            for token in tokens:
                if token not in seen:
                    seen.add(token)
                    vocab.append(token)
        return vocab

    def initialize(self, sentences):
        if self.BLOCK_SIZE < 1 or self.EMBED < 1:
            raise ValueError("block_size와 embed는 양수여야 합니다.")
        self.itos = self.build_vocab(sentences)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        target_device = self.device()
        self.net = self.build_net().to(target_device)
        return self

    def build_net(self):
        return SingleHeadLM(len(self.itos), self.EMBED, self.BLOCK_SIZE)

    def extra_metadata(self):
        return {}

    def restore_extra_metadata(self, meta):
        pass

    def make_windows(self, sentences):
        return _data.make_windows(self, sentences)

    def train(self, sentences, valid_sentences=None):
        return _training.fit(self, list(sentences),
                             list(valid_sentences) if valid_sentences is not None else None)

    def evaluate_rows(self, rows, top_k=5):
        """생성과 같은 슬라이딩 문맥. FLOOR PPL과 실제 어휘 coverage를 분리."""
        if top_k < 1:
            raise ValueError("top_k는 양수여야 합니다.")
        if self.net is None:
            raise ValueError("모델을 먼저 학습하거나 로드하세요.")
        n_all = n_cov = hit1 = hitk = 0
        total_log = 0.0
        was_training = self.net.training
        self.net.eval()
        try:
            with torch.no_grad():
                for x, y, known in self.batches(rows, self.EVAL_BATCH):
                    logits = self.net(x.to(self.device())).cpu()
                    active = y.ne(-100)
                    good = active & known
                    n = int(active.sum())
                    cov = int(good.sum())
                    n_all += n
                    n_cov += cov
                    total_log += (n - cov) * math.log(self.FLOOR)
                    if cov:
                        values = logits[good]
                        targets = y[good]
                        probs = values.softmax(-1).gather(1, targets[:, None]).squeeze(1)
                        total_log += probs.double().clamp_min(self.FLOOR).log().sum().item()
                        ranked = values.topk(min(top_k, len(self.itos) - 1), dim=-1).indices
                        hit1 += int((ranked[:, 0] == targets).sum())
                        hitk += int((ranked == targets[:, None]).any(-1).sum())
        finally:
            self.net.train(was_training)
        return {"ppl": math.exp(-total_log / n_all) if n_all else float("inf"),
                "top1": hit1 / n_all if n_all else 0.0,
                "topk": hitk / n_all if n_all else 0.0,
                "coverage": n_cov / n_all if n_all else 0.0,
                "n": n_all, "k": top_k}

    def evaluate(self, sentences, top_k=5):
        return self.evaluate_rows(self.make_windows(sentences), top_k)

    def perplexity(self, sentences):
        return self.evaluate(sentences)["ppl"]

    def accuracy(self, sentences, top_k=5):
        result = self.evaluate(sentences, top_k)
        return {k: v for k, v in result.items() if k != "ppl"}

    def _context_ids(self, recent):
        ids = [self.stoi.get(t, 0) for t in recent[-self.BLOCK_SIZE:]]
        return ids if ids and all(ids) else None

    def _probs(self, recent):
        if self.net is None:
            raise ValueError("모델을 먼저 학습하거나 로드하세요.")
        ids = self._context_ids(recent)
        if ids is None:
            return None
        was_training = self.net.training
        self.net.eval()
        try:
            with torch.no_grad():
                x = torch.tensor([ids], device=self.device())
                return self.net(x)[0, -1].softmax(-1).cpu()
        finally:
            self.net.train(was_training)

    def next_dist(self, recent):
        probs = self._probs(recent)
        return None if probs is None else {t: probs[i].item()
                                           for i, t in enumerate(self.itos) if i != 0}

    def token_prob(self, recent, token):
        probs = self._probs(recent)
        idx = self.stoi.get(token, 0)
        if probs is None or idx == 0:
            return self.FLOOR
        return max(self.FLOOR, probs[idx].item())

    def next_token(self, recent, temperature):
        dist = self.next_dist(recent)
        if dist is None:
            return None
        # log-space 보정으로 낮은 온도에서 모든 확률이 0으로 언더플로하는 것을 방지.
        if temperature <= 0.01:
            return max(dist, key=dist.get)
        scores = [(t, math.log(max(p, 1e-45)) / temperature) for t, p in dist.items()]
        peak = max(s for _, s in scores)
        weights = {t: math.exp(s - peak) for t, s in scores}
        return self.choose(weights, 1.0)

    def can_continue(self, recent):
        return self._context_ids(recent) is not None

    def generate(self, start_text, temperature=0.0, top_k=0, top_p=1.0):
        self._top_k, self._top_p = top_k, top_p
        recent = self.tokenize(start_text)
        for _ in range(self.MAX_LENGTH):
            nxt = self.next_token(recent, temperature)
            if nxt is None or self.is_end(nxt):
                break
            recent.append(nxt)
        return self.detokenize(recent)

    def save(self, model_path, vocab_path=None):
        """추론용 가중치+설정 저장. optimizer/RNG 재개는 v0.4의 범위입니다."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"version": self.MODEL_VERSION, "tokenizer": self.tokenizer_name(),
                "vocab": self.itos, "block_size": self.BLOCK_SIZE, "embed": self.EMBED,
                "max_length": self.MAX_LENGTH, "floor": self.FLOOR,
                "best_epoch": self.best_epoch, **self.extra_metadata()}
        torch.save({"format_version": 1, "meta": meta,
                    "state_dict": {k: v.detach().cpu() for k, v in self.net.state_dict().items()}}, path)
        Path(vocab_path or path.with_name("vocab.json")).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError(f"지원하지 않는 {self.MODEL_VERSION} 모델 형식입니다.")
        meta = checkpoint["meta"]
        if meta["version"] != self.MODEL_VERSION or meta["tokenizer"] != self.tokenizer_name():
            raise ValueError("모델 버전 또는 토크나이저가 일치하지 않습니다.")
        self.itos = meta["vocab"]
        if self.itos[0] != self.PAD or len(set(self.itos)) != len(self.itos):
            raise ValueError("어휘의 PAD 또는 중복 항목이 잘못되었습니다.")
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.BLOCK_SIZE, self.EMBED = meta["block_size"], meta["embed"]
        self.MAX_LENGTH, self.FLOOR = meta["max_length"], meta["floor"]
        self.best_epoch = meta["best_epoch"]
        self.restore_extra_metadata(meta)
        self.net = self.build_net().to(self.device())
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()
        return self


NGramLM = NeuralLM
Model = NeuralLM
DATA_PATH = str(ROOT / "data/pretrain/train.txt")
VALID_PATH = str(ROOT / "data/pretrain/valid.txt")
MODEL_PATH = str(HERE / "model.pt")
VOCAB_PATH = str(HERE / "vocab.json")
