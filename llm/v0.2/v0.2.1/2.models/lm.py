# -*- coding: utf-8 -*-
"""
lm.py  (v0.2.1)  -  문맥 확장 (block_size)  · 앞 N토큰 임베딩 concat = 진짜 Bengio 입력층

[v0.2.0 과 무엇이 다른가]
  v0.2.0 은 앞 **2토큰**(prev2, prev1)으로 고정이었어요. v0.2.1 은 이 문맥 길이를
  **하이퍼파라미터 `BLOCK_SIZE`(=N)** 로 열어, 앞 **N토큰**을 보고 다음을 맞힙니다.
  (Bengio 2003 의 입력층: 앞 N개 토큰 임베딩을 이어붙여 MLP 에 넣기.)

[바뀌는 것 — '문맥 길이 N' 하나]
  입력  : 앞 N토큰 인덱스 (B, N)
  모델  : 각 토큰 임베딩(E) → 이어붙임(N·E) → fc1 → tanh → fc2 → logits(V)
  나머지(임베딩·2층 MLP·정규화·저장/로드·대화·PPL)는 v0.2.0 을 **그대로 상속**.
  N=2 로 두면 v0.2.0 과 정확히 같아요 (일반화).

[문장 맨 앞의 빈자리 — <PAD>]
  N토큰이 다 안 차는 문장 앞부분은 <PAD>(어휘 0번)로 채웁니다. 학습·추론 동일.

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
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_VERSION_DIR)))  # 저장소 루트
_DATA_DIR = os.path.join(_ROOT, "data")                                  # 모든 버전 공용 데이터


def _load_prev_module(prev_version):
    """이전 버전 lm.py 모듈을 통째로 불러와요 (NGramLM 계보 재사용)."""
    group = prev_version.rsplit(".", 1)[0]                    # "v0.2.0" -> "v0.2"
    lmm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../llm 루트
    path = os.path.join(lmm_dir, group, prev_version, "2.models", "lm.py")
    spec = importlib.util.spec_from_file_location("lmmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_torch():
    if torch is None:
        raise SystemExit(
            "이 버전(v0.2.1)은 PyTorch 가 필요해요.\n"
            "  pip install torch\n"
            "(torch 를 설치할 수 있는 파이썬 환경에서 실행해 주세요.)"
        )


# v0.2.0(임베딩 · 2토큰 고정)을 물려받아, 문맥 길이를 N(BLOCK_SIZE)으로 일반화합니다.
_prev = _load_prev_module("v0.2.0")

_Module = nn.Module if torch is not None else object   # torch 없이도 이 파일이 로딩되게


# ---------- N토큰 문맥 임베딩 신경망 (Bengio 입력층) ----------
class BlockModel(_Module):
    """
    앞 N토큰 → 각 임베딩(E) → 이어붙임(N·E) → fc1 → tanh → fc2 → logits.
    v0.2.0 의 EmbeddingModel(2토큰 고정)을 **N토큰**으로 넓힌 것. N=2 면 동일.
      - forward(x): x 는 (B, N) 정수 인덱스
    """
    def __init__(self, vocab_size, hidden, embed, block_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.emb = nn.Embedding(vocab_size, embed)         # 임베딩 표 C: V×E
        self.fc1 = nn.Linear(block_size * embed, hidden)   # N개 임베딩 concat(N·E) → 은닉
        self.fc2 = nn.Linear(hidden, vocab_size)           # 은닉 → 출력(logits)

    def forward(self, x):                                  # x: (B, N)  앞 N토큰 인덱스
        e = self.emb(x)                                    # (B, N, E)  각 토큰 임베딩
        e = e.flatten(1)                                   # (B, N·E)   이어붙이기(concat)
        hidden = torch.tanh(self.fc1(e))                   # (B, H)
        return self.fc2(hidden)                            # logits (B, V)


class NeuralLM(_prev.NGramLM):
    # 하이퍼파라미터(BLOCK_SIZE / EMBED / HIDDEN / OPTIMIZER / LR / EPOCHS / BATCH_SIZE /
    #  SEED / WEIGHT_DECAY / INIT / LABEL_SMOOTHING)는 3.train/train.py 에서 설정해요.
    # (임베딩·정규화·저장/로드·대화·PPL 은 v0.2.0 까지를 상속, <PAD> 는 v0.1.4)

    def build_net(self, vocab_size, hidden):
        """이 버전의 신경망 = N토큰 임베딩 MLP. (은닉은 인자, 임베딩·문맥길이는 self.EMBED/BLOCK_SIZE)"""
        return BlockModel(vocab_size, hidden, self.EMBED, self.BLOCK_SIZE)

    # ---------- 데이터: 앞 N토큰 → 다음 ----------
    def make_pairs(self, sentences):
        """
        문장을 (앞 N토큰 → 다음) 짝으로. 앞이 모자라면 <PAD> 로 채워요.
        예) ids=[a,b,c], N=3 (PAD=p) →
            (p,p,a)->b, (p,a,b)->c, (a,b,c)-><END>
        (N=2 면 v0.2.0 과 동일)
        """
        pad = self.stoi[self.PAD]
        N = self.BLOCK_SIZE
        xs, ys = [], []
        for s in sentences:
            ids = [self.stoi[t] for t in self.prepare(self.tokenize(s))]
            for j in range(1, len(ids)):
                ctx = [ids[k] if k >= 0 else pad for k in range(j - N, j)]  # 앞 N토큰(오래된→최근)
                xs.append(ctx)
                ys.append(ids[j])
        return xs, ys

    # ---------- 추론 문맥도 앞 N토큰으로 ----------
    def _context_tensor(self, recent):
        """recent 의 마지막 N토큰을 (1, N) 텐서로. 앞이 없거나 어휘 밖이면 None/<PAD>."""
        if not recent or recent[-1] not in self.stoi:
            return None
        pad = self.stoi[self.PAD]
        N = self.BLOCK_SIZE
        ctx = []
        for k in range(len(recent) - N, len(recent)):
            if k < 0:
                ctx.append(pad)
            else:
                ctx.append(self.stoi.get(recent[k], pad))
        return torch.tensor([ctx], dtype=torch.long)   # (1, N)

    # ---------- 불러오기: 저장된 모양에서 N·E·H 복원 ----------
    def load(self, model_path):
        _require_torch()
        vocab_path = os.path.join(os.path.dirname(model_path), "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.itos = meta["vocab"]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        V = len(self.itos)

        state = torch.load(model_path, map_location="cpu")
        self.EMBED = state["emb.weight"].shape[1]                 # 임베딩 차원 E
        self.BLOCK_SIZE = state["fc1.weight"].shape[1] // self.EMBED   # 문맥 길이 N = (N·E)/E
        hidden = state["fc1.weight"].shape[0]                     # 은닉 H
        self.net = self.build_net(V, hidden)
        self.net.load_state_dict(state)
        self.net.eval()
        return self


# web_service / 상속 사슬이 module.NGramLM 을 찾으므로 노출.
NGramLM = NeuralLM


class Model(NeuralLM):
    pass


DATA_PATH = os.path.join(_DATA_DIR, "data.txt")      # 학습용
VALID_PATH = os.path.join(_DATA_DIR, "valid.txt")    # 검증용
MODEL_PATH = os.path.join(_HERE, "model.pt")                      # 가중치 (PyTorch 표준)
VOCAB_PATH = os.path.join(_HERE, "vocab.json")                    # 어휘 (0번=<PAD>)
