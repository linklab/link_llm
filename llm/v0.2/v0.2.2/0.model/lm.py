# -*- coding: utf-8 -*-
"""
lm.py  (v0.2.2)  -  초기화 · 정규화 (깊은 MLP 안정 · 과적합 완화)

[왜 이 버전인가 — v0.2.1 의 문제]
  v0.2.1(문맥 N=3)은 조기 종료를 끈 채 끝까지 학습해 **과적합**했어요(학습 2.56 vs 검증 62.98).
  v0.2.2 는 '올바른 초기화 + 정규화 + 조기 종료'라는 **표준 도구 세트**를 갖춰 이를 잡습니다.

[무엇을 더하나 — 정규화·초기화 도구 세트]
  ① 올바른 **초기화** (Kaiming/Xavier, tanh 이득 반영) : 층 크기에 맞춘 시작값 → 안정적 학습.
  ② **BatchNorm** (토글 USE_BN) : fc1 뒤·tanh 앞에 넣어 은닉값 분포를 배치마다 정규화.
  ③ **weight_decay(L2)** + **조기 종료**(검증 PPL 최고점 복원) 로 과적합 억제.

[정직한 실험 결과 — 규모의 교훈]
  이 데이터(282문장)에선 **BatchNorm 을 켜면 오히려 검증 PPL 이 살짝 올라가요**
  (BN on ≈ 41, BN off ≈ 38). BatchNorm 은 **깊고 큰 모델**에서 빛나는 기법이라, 작은 모델엔
  이득이 적어요. 그래서 기본값은 **USE_BN=False** (Kaiming init + weight_decay + 조기 종료가 핵심).
  USE_BN=True 로 바꿔 직접 비교해 보세요 — "모든 기법이 모든 규모에서 통하진 않는다".

[구조]  (앞 N토큰) → 임베딩 concat(N·E) → fc1 → [BatchNorm(선택)] → tanh → fc2 → logits(V)

[그대로인 것]  임베딩·문맥(block_size)·데이터·옵티마이저·조기 종료·대화·PPL. (build_net·init_model·load 만 조정)

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
    """이전 버전 lm.py 모듈을 통째로 불러와요 (NGramLM 계보 재사용)."""
    group = prev_version.rsplit(".", 1)[0]                    # "v0.2.1" -> "v0.2"
    llm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../llm 루트
    path = os.path.join(llm_dir, group, prev_version, "0.model", "lm.py")
    spec = importlib.util.spec_from_file_location("llmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_torch():
    if torch is None:
        raise SystemExit("이 버전(v0.2.2)은 PyTorch 가 필요해요.\n  pip install torch")


# v0.2.1(N토큰 임베딩 MLP)을 물려받아, 초기화·정규화 도구만 더합니다.
_prev = _load_prev_module("v0.2.1")

_Module = nn.Module if torch is not None else object   # torch 없이도 이 파일이 로딩되게


# ---------- N토큰 임베딩 MLP + (선택) BatchNorm ----------
class NormModel(_Module):
    """
    앞 N토큰 → 임베딩 concat(N·E) → fc1 → [BatchNorm] → tanh → fc2 → logits.
    use_bn=True 면 fc1 뒤에 BatchNorm1d 를 끼워요 (그때 fc1 은 bias=False — BN 의 beta 가 대신).
    """
    def __init__(self, vocab_size, hidden, embed, block_size, use_bn=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.emb = nn.Embedding(vocab_size, embed)
        self.fc1 = nn.Linear(block_size * embed, hidden, bias=not use_bn)
        self.bn = nn.BatchNorm1d(hidden) if use_bn else None
        self.fc2 = nn.Linear(hidden, vocab_size)

    def forward(self, x):                              # x: (B, N)
        z = self.fc1(self.emb(x).flatten(1))           # (B, hidden)
        if self.bn is not None:
            z = self.bn(z)                             # 배치 정규화 (선택)
        return self.fc2(torch.tanh(z))                 # logits (B, V)


class NeuralLM(_prev.NGramLM):
    # 하이퍼파라미터(USE_BN / INIT / WEIGHT_DECAY / BLOCK_SIZE / EMBED / HIDDEN / OPTIMIZER / LR /
    #  EPOCHS / BATCH_SIZE / SEED / LABEL_SMOOTHING / EARLY_STOPPING·PATIENCE)는 1.train/train.py 에서.
    USE_BN = False        # BatchNorm 사용 여부 (이 작은 데이터엔 꺼두는 게 나아요)

    def build_net(self, vocab_size, hidden):
        return NormModel(vocab_size, hidden, self.EMBED, self.BLOCK_SIZE, use_bn=self.USE_BN)

    def init_model(self, model):
        """
        INIT 선택:  "kaiming"(기본)=tanh 이득 반영한 Xavier/Kaiming 계열 초기화 /
                     "zeros"=마지막 층 0(순진, 비교용) / "default"=PyTorch 기본값.
        """
        if self.INIT == "zeros":
            nn.init.zeros_(model.fc2.weight); nn.init.zeros_(model.fc2.bias); return
        if self.INIT == "default":
            return
        gain = nn.init.calculate_gain("tanh")          # ≈ 1.667
        nn.init.xavier_normal_(model.fc1.weight, gain=gain)
        nn.init.xavier_normal_(model.fc2.weight); nn.init.zeros_(model.fc2.bias)

    def load(self, model_path):
        """model.pt + vocab.json 로 복원 (저장된 가중치에서 E·N·H·BatchNorm 유무까지 되살림)."""
        _require_torch()
        vocab_path = os.path.join(os.path.dirname(model_path), "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.itos = meta["vocab"]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        V = len(self.itos)

        state = torch.load(model_path, map_location="cpu")
        self.EMBED = state["emb.weight"].shape[1]
        self.BLOCK_SIZE = state["fc1.weight"].shape[1] // self.EMBED
        self.USE_BN = "bn.weight" in state             # 저장 당시 BatchNorm 여부
        hidden = state["fc1.weight"].shape[0]
        self.net = self.build_net(V, hidden)
        self.net.load_state_dict(state)
        self.net.eval()
        return self


# web_service / 상속 사슬이 module.NGramLM 을 찾으므로 노출.
NGramLM = NeuralLM


class Model(NeuralLM):
    pass


_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_VERSION_DIR)))  # 저장소 루트
_DATA_DIR = os.path.join(_ROOT, "data")                                  # 모든 버전 공용 데이터
DATA_PATH = os.path.join(_DATA_DIR, "data.txt")      # 학습용
VALID_PATH = os.path.join(_DATA_DIR, "valid.txt")    # 검증용
MODEL_PATH = os.path.join(_HERE, "model.pt")         # 가중치 (PyTorch 표준)
VOCAB_PATH = os.path.join(_HERE, "vocab.json")       # 어휘 (0번=<PAD>)
