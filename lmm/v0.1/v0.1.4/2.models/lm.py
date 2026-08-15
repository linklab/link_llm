# -*- coding: utf-8 -*-
"""
lm.py  (v0.1.4)  -  2토큰 문맥 신경망 (드디어 카운트와 '같은 문맥'으로 공정한 대결)

[왜 이 버전이 필요한가 — v0.1.3 까지의 한계]
  v0.1.0~v0.1.3 신경망은 **앞 '1토큰'만** 보고 다음을 맞혔어요 (W[prev]).
  그런데 비교 상대인 v0.0.9(카운트)는 **앞 '2토큰' + 백오프**(ORDERS=[1,2])를 써요.
  즉 카운트가 문맥을 더 많이 봐서 유리한 '불공정 대결'이었어요.
  (그래서 신경망이 검증 PPL 에서 카운트에 졌던 거예요 — 실력이 아니라 문맥 길이 차이.)

[이 버전이 바꾸는 것 — 딱 하나, '문맥 길이']
  신경망도 **앞 2토큰(prev2, prev1)** 을 보고 다음을 맞히게 합니다.
    입력  : 두 개의 one-hot 를 '이어붙여(concat)' 2V 차원으로
    모델  : nn.Linear(2V -> V)  (여전히 은닉층·임베딩 없음 — 딱 한 층)
    출력  : 다음 토큰 logits (V)
  이제 v0.0.9 와 '같은 자'로 재는 완전히 공정한 대결이 됩니다.
  게다가 신경망엔 softmax(+label smoothing)라는 부드러움이 있어, 카운트의
  FLOOR 절벽(처음 보는 조합에 큰 벌점)을 피할 수 있어요 → 여기서 이길 여지가 생겨요.

[문장 맨 앞은? — <PAD>]
  두 번째 토큰을 맞힐 땐 '앞앞 토큰'이 없어요. 그 자리는 <PAD>(빈자리)로 채웁니다.
  <PAD> 를 어휘 0번에 넣어, 학습·추론이 똑같이 '없으면 PAD'로 처리해요.

[그대로인 것]  정규화·초기화(v0.1.3) · Dataset/DataLoader(v0.1.1) · torch.optim(v0.1.2) ·
  토크나이저 · <END> · 대화(chat) · 샘플링 · 퍼플렉서티. 저장 형식만 W -> W2 로 넓어져요.

[주의]  PyTorch 필요.   pip install torch
"""

import os
import json
import importlib.util

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:
    torch = None
    F = None
    nn = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERSION_DIR = os.path.dirname(_HERE)


def _load_prev_module(prev_version):
    """이전 버전 lm.py 모듈을 통째로 불러와요 (NGramLM · build_dataloader 재사용)."""
    group = prev_version.rsplit(".", 1)[0]                    # "v0.1.3" -> "v0.1"
    lmm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../lmm 루트
    path = os.path.join(lmm_dir, group, prev_version, "2.models", "lm.py")
    spec = importlib.util.spec_from_file_location("lmmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_torch():
    if torch is None:
        raise SystemExit(
            "이 버전(v0.1.4)은 PyTorch 가 필요해요.\n"
            "  pip install torch\n"
            "(torch 를 설치할 수 있는 파이썬 환경에서 실행해 주세요.)"
        )


# v0.1.3 을 물려받아 정규화·초기화(make_optimizer/init_model)를 그대로 재사용해요.
_prev = _load_prev_module("v0.1.3")
build_dataloader = _prev.build_dataloader   # (앞, 다음) 텐서 -> DataLoader (v0.1.1 것)

_Module = nn.Module if torch is not None else object   # torch 없이도 이 파일이 로딩되게


# ---------- 2토큰 문맥 신경망 (one-hot 두 개를 이어붙여 한 층에) ----------
class ContextModel(_Module):
    """
    앞 2토큰(prev2, prev1) -> 각각 one-hot -> 이어붙여(2V) -> nn.Linear -> 다음 토큰 logits.
    v0.1.0 의 BigramModel(1토큰, V->V)을 '문맥 2토큰'으로 넓힌 것 (2V->V). 여전히 한 층.
      - forward(x2): x2 는 (B, 2) = [[prev2, prev1], ...]
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        # 입력이 2V(두 one-hot 이어붙임)라는 점만 빼면 BigramModel 과 똑같이 bias 없는 한 층.
        self.linear = nn.Linear(2 * vocab_size, vocab_size, bias=False)

    def forward(self, x2):                              # x2: (B, 2)  [prev2, prev1]
        oh2 = F.one_hot(x2[:, 0], num_classes=self.vocab_size).float()   # 앞앞 (B, V)
        oh1 = F.one_hot(x2[:, 1], num_classes=self.vocab_size).float()   # 앞   (B, V)
        x = torch.cat([oh2, oh1], dim=1)               # 이어붙이기 (B, 2V)
        return self.linear(x)                          # logits (B, V)


# v0.1.3(정규화·초기화)을 물려받아, '문맥 1토큰 -> 2토큰' 으로 모델만 넓힙니다.
class NeuralLM(_prev.NGramLM):
    # 하이퍼파라미터(OPTIMIZER / LR / EPOCHS / BATCH_SIZE / SEED / WEIGHT_DECAY /
    #  INIT / LABEL_SMOOTHING)는 3.train/train.py 에서 설정해요.
    PAD = "<PAD>"        # 문장 맨 앞 '앞앞 토큰' 빈자리를 채우는 특수 토큰 (어휘 0번)

    # ---------- 어휘 / 데이터 준비 (2토큰 문맥용으로 재정의) ----------
    def build_vocab(self, sentences):
        """v0.1.0 의 어휘 앞에 <PAD> 를 0번으로 추가해요 (빈 앞앞 자리를 가리킬 인덱스)."""
        return [self.PAD] + super().build_vocab(sentences)

    def make_pairs(self, sentences):
        """
        문장을 ((앞앞, 앞) -> 다음) 짝으로 바꿉니다. 앞앞이 없으면 <PAD>.

        예) ids = [0, 1, 2, 3]  (PAD 는 p 로 표기)
          맞힐 대상:  1        2        3
          문맥(앞앞,앞): (p,0)  (0,1)   (1,2)
          => xs2 = [[p,0], [0,1], [1,2]] ,  ys = [1, 2, 3]
        (v0.0.9 가 앞 2토큰을 쓰고 없으면 1토큰으로 백오프하던 것과 같은 '2토큰 문맥' 정신.
         신경망은 백오프 대신 <PAD> 자리로 '앞앞 없음'을 학습해요.)
        """
        pad = self.stoi[self.PAD]
        xs2, ys = [], []
        for s in sentences:
            ids = [self.stoi[t] for t in self.prepare(self.tokenize(s))]
            for j in range(1, len(ids)):
                prev1 = ids[j - 1]
                prev2 = ids[j - 2] if j >= 2 else pad
                xs2.append([prev2, prev1])
                ys.append(ids[j])
        return xs2, ys

    # ---------- 학습 (2토큰 문맥 모델) ----------
    def train(self, sentences):
        _require_torch()
        # 1) 어휘(+PAD) 와 ((앞앞,앞) -> 다음) 짝
        self.itos = self.build_vocab(sentences)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        V = len(self.itos)

        xs_list, ys_list = self.make_pairs(sentences)
        xs = torch.tensor(xs_list, dtype=torch.long)    # (N, 2)
        ys = torch.tensor(ys_list, dtype=torch.long)    # (N,)

        # 2) 데이터로더 (v0.1.1 재사용 — 입력이 (2,) 벡터라도 그대로 배치돼요)
        loader = build_dataloader(xs, ys, self.BATCH_SIZE, shuffle=True)

        # 3) 모델 + 초기화 + 옵티마이저 (v0.1.3 의 정규화·초기화 그대로)
        torch.manual_seed(self.SEED)
        model = ContextModel(V)
        self.init_model(model)                    # zeros_ 등 (model.linear 대상)
        optimizer = self.make_optimizer(model)    # weight_decay 포함

        self.losses = []
        print(f"  학습 시작(2토큰 문맥): 어휘 {V}개, 짝 {len(xs_list)}개, epochs {self.EPOCHS}, "
              f"batch {self.BATCH_SIZE}, optim={self.OPTIMIZER}, lr {self.LR}, "
              f"weight_decay {self.WEIGHT_DECAY}, init={self.INIT}, label_smoothing {self.LABEL_SMOOTHING}")

        for epoch in range(1, self.EPOCHS + 1):
            total, n_batches = 0.0, 0
            for batch in loader:
                logits = model(batch["input"])            # (B, V)  — batch["input"]: (B, 2)
                loss = F.cross_entropy(logits, batch["target"],
                                       label_smoothing=self.LABEL_SMOOTHING)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
                n_batches += 1
            avg = total / n_batches
            self.losses.append(avg)
            if epoch == 1 or epoch % 5 == 0:
                print(f"  epoch {epoch:3d}/{self.EPOCHS}   avg loss {avg:.4f}")

        # 추론/저장이 쓰는 self.W2 로 보관. shape (V, 2V) — 앞앞 V칸 + 앞 V칸.
        self.W2 = model.linear.weight.detach().contiguous()
        return self.W2

    # ---------- 저장 / 불러오기 (W -> W2 로 형식만 넓힘) ----------
    def to_dict(self, W2):
        return {
            "type": "neural_context2",       # 2토큰 문맥 신경망
            "tokenizer": self.tokenizer_name(),
            "vocab": self.itos,              # 0번이 <PAD>
            "W2": W2.tolist(),               # (V, 2V)
        }

    def save(self, W2, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(W2), f, ensure_ascii=False)

    def load(self, path):
        _require_torch()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.model = data
        self.itos = data["vocab"]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.W2 = torch.tensor(data["W2"], dtype=torch.float32)   # (V, 2V)
        return self

    # ---------- 확률 엔진: 2토큰 문맥으로 logits ----------
    def _probs_for2(self, prev2_idx, prev1_idx):
        """
        앞 2토큰 다음에 올 토큰들의 확률 = softmax(logits).
        one-hot 두 개를 이어붙였으므로 logits[j] = W2[j, prev2] + W2[j, V+prev1]
        (앞앞은 앞쪽 V칸, 앞은 뒤쪽 V칸에 해당).
        """
        V = self.W2.shape[0]
        logits = self.W2[:, prev2_idx] + self.W2[:, V + prev1_idx]
        return torch.softmax(logits, dim=0)

    def _ctx_indices(self, recent):
        """recent 에서 (앞앞, 앞) 인덱스를 뽑아요. 앞이 없거나 어휘 밖이면 None(이어갈 수 없음)."""
        if not recent or recent[-1] not in self.stoi:
            return None
        pad = self.stoi[self.PAD]
        prev1_idx = self.stoi[recent[-1]]
        prev2 = recent[-2] if len(recent) >= 2 else None
        prev2_idx = self.stoi.get(prev2, pad)      # 없거나 어휘 밖이면 <PAD>
        return prev2_idx, prev1_idx

    def token_prob(self, recent, token):
        """퍼플렉서티(v0.0.9)가 부르는 함수 — 2토큰 문맥의 신경망 확률로 답합니다."""
        if token not in self.stoi:
            return self.FLOOR
        ctx = self._ctx_indices(recent)
        if ctx is None:
            return self.FLOOR
        return float(self._probs_for2(*ctx)[self.stoi[token]])

    def next_token(self, recent, temperature):
        """다음 토큰을 2토큰 문맥의 신경망 분포에서 뽑아요 (샘플링은 choose() 재사용)."""
        ctx = self._ctx_indices(recent)
        if ctx is None:
            return None
        probs = self._probs_for2(*ctx)
        candidates = {self.itos[j]: float(probs[j]) for j in range(len(self.itos))}
        return self.choose(candidates, temperature)

    def can_continue(self, recent):
        return bool(recent) and recent[-1] in self.stoi


# web_service / 상속 사슬이 module.NGramLM 을 찾으므로 노출.
NGramLM = NeuralLM


class Model(NeuralLM):
    pass


DATA_PATH = os.path.join(_VERSION_DIR, "1.data", "data.txt")      # 학습용
VALID_PATH = os.path.join(_VERSION_DIR, "1.data", "valid.txt")    # 검증용
MODEL_PATH = os.path.join(_HERE, "model.json")
