# -*- coding: utf-8 -*-
"""
lm.py  (v0.1.4)  -  2토큰 문맥 신경망 (2층 MLP)  (카운트와 '같은 문맥'으로 공정한 대결)

[왜 이 버전인가]
  v0.1.0~v0.1.3 신경망은 앞 '1토큰'만 봤는데, 비교 상대 v0.0.9(카운트)는 앞 '2토큰'을 써요.
  문맥이 짧아 불리했어요. v0.1.4 는 신경망도 **앞 2토큰(prev2, prev1)** 을 보게 합니다.

[이 버전이 바꾸는 것 — 딱 '문맥 길이'와 그에 맞춘 입력]
  입력  : 두 one-hot 를 이어붙여 2V 차원
  모델  : 2V → [fc1] → 은닉(H) → tanh → [fc2] → V   (여전히 2개 층, v0.1.0 과 같은 구조에 입력만 2배)
  나머지(정규화·초기화·DataLoader·옵티마이저·저장/로드·대화·PPL)는 v0.1.3 까지를 **그대로 상속**.

[문장 맨 앞은? — <PAD>]
  두 번째 토큰을 맞힐 땐 '앞앞 토큰'이 없어요. 그 자리는 <PAD>(어휘 0번)로 채웁니다.

[주의]  PyTorch 필요.   pip install torch
"""

import os
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
    group = prev_version.rsplit(".", 1)[0]                    # "v0.1.3" -> "v0.1"
    llm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../llm 루트
    path = os.path.join(llm_dir, group, prev_version, "0.model", "lm.py")
    spec = importlib.util.spec_from_file_location("llmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# v0.1.3(정규화·초기화)을 물려받아, 모델의 '문맥 1토큰 → 2토큰' 만 바꿉니다.
_prev = _load_prev_module("v0.1.3")

_Module = nn.Module if torch is not None else object   # torch 없이도 이 파일이 로딩되게


# ---------- 2토큰 문맥 신경망 (2층: 이어붙인 one-hot → 은닉(tanh) → 출력) ----------
class ContextModel(_Module):
    """
    앞 2토큰(prev2, prev1) → 각각 one-hot → 이어붙임(2V) → fc1 → tanh → fc2 → logits.
    v0.1.0 의 BigramModel(입력 V)을 '문맥 2토큰'(입력 2V)으로 넓힌 것. 층 수(2개)는 같아요.
      - forward(x2): x2 는 (B, 2) = [[prev2, prev1], ...]
    """
    def __init__(self, vocab_size, hidden):
        super().__init__()
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(2 * vocab_size, hidden)   # 1층: 입력(2V) → 은닉
        self.fc2 = nn.Linear(hidden, vocab_size)       # 2층: 은닉 → 출력(logits)

    def forward(self, x2):                              # x2: (B, 2)  [prev2, prev1]
        oh2 = F.one_hot(x2[:, 0], num_classes=self.vocab_size).float()   # 앞앞 (B, V)
        oh1 = F.one_hot(x2[:, 1], num_classes=self.vocab_size).float()   # 앞   (B, V)
        hidden = torch.tanh(self.fc1(torch.cat([oh2, oh1], dim=1)))      # (B, 2V) → (B, H)
        return self.fc2(hidden)                                          # logits (B, V)


class NeuralLM(_prev.NGramLM):
    # 하이퍼파라미터(OPTIMIZER / HIDDEN / LR / EPOCHS / BATCH_SIZE / SEED / WEIGHT_DECAY /
    #  INIT / LABEL_SMOOTHING)는 1.train/train.py 에서 설정해요.
    PAD = "<PAD>"        # 문장 맨 앞 '앞앞 토큰' 빈자리를 채우는 특수 토큰 (어휘 0번)

    # ---------- 신경망을 2토큰용으로 (build_net override) ----------
    def build_net(self, vocab_size, hidden):
        return ContextModel(vocab_size, hidden)

    # ---------- 어휘 / 데이터 준비 (2토큰 문맥용으로 재정의) ----------
    def build_vocab(self, sentences):
        """v0.1.0 의 어휘 앞에 <PAD> 를 0번으로 추가해요 (빈 앞앞 자리를 가리킬 인덱스)."""
        return [self.PAD] + super().build_vocab(sentences)

    def make_pairs(self, sentences):
        """
        문장을 ((앞앞, 앞) -> 다음) 짝으로 바꿉니다. 앞앞이 없으면 <PAD>.
        예) ids=[<사용자>,안녕,<봇>] (PAD=p) →
            (p,<사용자>)->안녕, (<사용자>,안녕)-><봇>, ...
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

    # ---------- 추론 문맥도 2토큰으로 (_context_ids override) ----------
    def _context_ids(self, recent):
        """recent 에서 (앞앞, 앞) 두 개를 골라요. 앞이 없거나 어휘 밖이면 None.
        (텐서로 감싸는 일·여러 개를 한 번에 채점하는 일은 v0.1.0 이 알아서 합니다.)"""
        if not recent or recent[-1] not in self.stoi:
            return None
        pad = self.stoi[self.PAD]
        prev1 = self.stoi[recent[-1]]
        prev2 = self.stoi.get(recent[-2], pad) if len(recent) >= 2 else pad
        return [prev2, prev1]                                            # 2개짜리 목록


# web_service / 상속 사슬이 module.NGramLM 을 찾으므로 노출.
NGramLM = NeuralLM


class Model(NeuralLM):
    pass


DATA_PATH = os.path.join(_DATA_DIR, "pretrain", "train.txt")      # 학습용
VALID_PATH = os.path.join(_DATA_DIR, "pretrain", "valid.txt")    # 검증용
MODEL_PATH = os.path.join(_HERE, "model.pt")                      # 가중치 (PyTorch 표준)
VOCAB_PATH = os.path.join(_HERE, "vocab.json")                    # 어휘 (0번=<PAD>)
