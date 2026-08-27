# -*- coding: utf-8 -*-
"""
lm.py  (v0.2.0)  -  임베딩 도입 (one-hot → nn.Embedding)  · 신경망 시대 2막의 시작

[v0.1.x 의 한계와 이 버전의 목적]
  v0.1.x 신경망은 토큰을 **one-hot** 으로 넣었어요 — 토큰마다 '독립된 칸'이라
  "안녕 다음"에서 배운 걸 비슷한 "반가워 다음"에 못 써먹어요(문맥끼리 공유 불가).
  v0.2.0 은 토큰을 **임베딩 벡터**(작은 연속 벡터)로 바꿉니다. 비슷한 토큰이 비슷한
  벡터를 갖게 학습돼, **문맥끼리 강점을 공유**해요 → 처음 보는 조합도 부드럽게 일반화.

[바뀌는 것은 딱 '입력 표현' 하나]
  v0.1.5 그대로 : 앞 2토큰 문맥 · 2층 MLP(은닉+tanh) · 정규화/초기화 · DataLoader ·
                  옵티마이저 · 저장/로드(model.pt + vocab.json) · 대화 · 퍼플렉서티.
  이 버전만    : one-hot(2V) → **임베딩 concat(2E)**. 즉 입력 차원이 확 작아지고(공유),
                  임베딩 표 C(V×E)를 함께 학습해요.

[핵심 통찰]  nn.Embedding(V, E)(idx)  ==  F.one_hot(idx, V).float() @ C
  '원-핫 곱하기 가중치'가 곧 '표에서 한 줄 꺼내기(lookup)'예요. one-hot 의 첫 층을
  임베딩 lookup 으로 바꾼 것 — 결과는 같지만 **E ≪ V 로 압축·공유**된다는 게 핵심.

[구조]  (앞앞, 앞) → 임베딩 lookup → concat(2E) → fc1 → tanh → fc2 → logits(V)  (Bengio 2003)

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
    """이전 버전 lm.py 모듈을 통째로 불러와요 (NGramLM 계보 재사용, 그룹이 달라도 경로로)."""
    group = prev_version.rsplit(".", 1)[0]                    # "v0.1.5" -> "v0.1"
    llm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../llm 루트
    path = os.path.join(llm_dir, group, prev_version, "2.models", "lm.py")
    spec = importlib.util.spec_from_file_location("llmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_torch():
    if torch is None:
        raise SystemExit(
            "이 버전(v0.2.0)은 PyTorch 가 필요해요.\n"
            "  pip install torch\n"
            "(torch 를 설치할 수 있는 파이썬 환경에서 실행해 주세요.)"
        )


# v0.1.5(2토큰 문맥 · 2층 MLP · 정규화)를 물려받아, '입력 표현'만 임베딩으로 바꿉니다.
_prev = _load_prev_module("v0.1.5")

_Module = nn.Module if torch is not None else object   # torch 없이도 이 파일이 로딩되게


# ---------- 임베딩 신경망 (Bengio 2003 스타일) ----------
class EmbeddingModel(_Module):
    """
    앞 2토큰 → 임베딩 lookup(각 E차원) → 이어붙임(2E) → fc1 → tanh → fc2 → logits.
    v0.1.4 의 ContextModel 과 층 구성(2층 MLP)은 같고, **입력이 one-hot(2V) → 임베딩(2E)** 로만 바뀜.
      - emb : 어휘표 C(V×E). emb(idx) = one-hot(idx) @ C 와 동치인 'lookup'.
      - forward(x2): x2 (B, 2) = [[prev2, prev1], ...]
    """
    def __init__(self, vocab_size, hidden, embed):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, embed)     # 임베딩 표 C: V×E (함께 학습)
        self.fc1 = nn.Linear(2 * embed, hidden)        # 1층: 두 임베딩 concat(2E) → 은닉
        self.fc2 = nn.Linear(hidden, vocab_size)       # 2층: 은닉 → 출력(logits)

    def forward(self, x2):                              # x2: (B, 2)  [prev2, prev1]
        e2 = self.emb(x2[:, 0])                         # 앞앞 임베딩 (B, E)
        e1 = self.emb(x2[:, 1])                         # 앞   임베딩 (B, E)
        hidden = torch.tanh(self.fc1(torch.cat([e2, e1], dim=1)))   # (B, 2E) → (B, H)
        return self.fc2(hidden)                         # logits (B, V)


class NeuralLM(_prev.NGramLM):
    # 하이퍼파라미터(EMBED / HIDDEN / OPTIMIZER / LR / EPOCHS / BATCH_SIZE / SEED /
    #  WEIGHT_DECAY / INIT / LABEL_SMOOTHING)는 3.train/train.py 에서 설정해요.
    # (문맥 2토큰 · make_pairs · _context_tensor · <PAD> 는 v0.1.4 것을 그대로 상속)

    def build_net(self, vocab_size, hidden):
        """이 버전의 신경망 = 임베딩 MLP. (은닉 크기는 인자로, 임베딩 크기는 self.EMBED)"""
        return EmbeddingModel(vocab_size, hidden, self.EMBED)

    def load(self, model_path):
        """model.pt + vocab.json 로 복원. 저장된 가중치에서 임베딩·은닉 크기를 되살려요."""
        _require_torch()
        vocab_path = os.path.join(os.path.dirname(model_path), "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.itos = meta["vocab"]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        V = len(self.itos)

        state = torch.load(model_path, map_location="cpu")
        self.EMBED = state["emb.weight"].shape[1]     # 임베딩 차원 복원 (V×E 의 E)
        hidden = state["fc1.weight"].shape[0]         # 은닉 크기 복원
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
