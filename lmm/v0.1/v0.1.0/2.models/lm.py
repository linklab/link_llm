# -*- coding: utf-8 -*-
"""
lm.py  (v0.1.0)  -  신경망 언어모델 (2층 MLP) + PyTorch autograd  ('세기'가 아니라 '학습')

[개수 세기 시대(v0.0.x)와 무엇이 다른가]
  지금까지는 "앞 토큰 다음에 무엇이 몇 번 나왔나"를 **세어서** 확률을 만들었어요.
  v0.1.0 은 그 확률을 **학습**합니다. 앞 토큰(one-hot)을 입력으로 받아
    입력(V) → [1층 nn.Linear] → 은닉(H) → tanh → [2층 nn.Linear] → 출력 logits(V)
  로 흐르는 **2개 층짜리 작은 신경망(MLP)** 을 두고, 경사하강법으로 가중치를 고쳐가요.
  미분(기울기)은 PyTorch **autograd**(`loss.backward()`)가 대신 계산합니다.

[인터페이스는 그대로 — '확률 엔진'만 교체]
  토크나이저 · <END> · 대화(chat) · 온도/top-k·top-p 샘플링 · 퍼플렉서티(PPL)는
  v0.0.x 것을 **그대로 물려받고**, "다음 토큰 확률을 어디서 얻느냐"(개수 표 → 신경망)만 바꿔요.

[저장 방식 — PyTorch 표준]
  가중치는 `torch.save(net.state_dict(), model.pt)` 로 저장하고, 어휘는 `vocab.json`
  (토크나이저 + 어휘 목록)에 따로 담아요. 불러올 때 저장된 가중치 모양에서 은닉 크기를
  복원해 신경망을 다시 만들고 `load_state_dict` 로 채웁니다.

[주의]  이 버전부터 PyTorch 가 필요합니다.   pip install torch
"""

import os
import json
import importlib.util

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:          # torch 가 없으면 학습/생성 시점에 안내 (모듈 로딩 자체는 되게)
    torch = None
    F = None
    nn = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERSION_DIR = os.path.dirname(_HERE)


def _load_prev(prev_version):
    group = prev_version.rsplit(".", 1)[0]                    # "v0.0.9" -> "v0.0" (마이너 그룹)
    lmm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../lmm 루트
    path = os.path.join(lmm_dir, group, prev_version, "2.models", "lm.py")
    spec = importlib.util.spec_from_file_location("lmmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NGramLM


def _require_torch():
    if torch is None:
        raise SystemExit(
            "이 버전(v0.1.0)은 PyTorch 가 필요해요.\n"
            "  pip install torch\n"
            "(torch 를 설치할 수 있는 파이썬 환경에서 실행해 주세요.)"
        )


# ---------- 신경망 모델 (2개 층: 입력 → 은닉(tanh) → 출력) ----------
_Module = nn.Module if torch is not None else object   # torch 없이도 이 파일이 로딩되게


class BigramModel(_Module):
    """
    앞 토큰(one-hot) → 은닉층(tanh) → 다음 토큰 점수(logits) 를 내는 작은 신경망(MLP).
    은닉층 없는 한 층이던 것을 **2개 층**으로 늘렸어요 (fc1: V→H, fc2: H→V, 사이에 tanh).
      - __init__ : 두 개의 nn.Linear 층을 정의
      - forward  : x(앞 토큰 인덱스) → one-hot → fc1 → tanh → fc2 → logits
    (문맥 확장·임베딩은 v0.1.4 / v0.2.x 에서.)
    """
    def __init__(self, vocab_size, hidden):
        super().__init__()
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(vocab_size, hidden)   # 1층: 입력(one-hot) → 은닉
        self.fc2 = nn.Linear(hidden, vocab_size)   # 2층: 은닉 → 출력(logits)

    def forward(self, x):                           # x: 앞 토큰 인덱스 (B,)
        onehot = F.one_hot(x, num_classes=self.vocab_size).float()   # (B, V)
        hidden = torch.tanh(self.fc1(onehot))       # 은닉층 + 비선형 (B, H)
        return self.fc2(hidden)                      # logits (B, V)


# v0.0.9 의 NGramLM(개수 세기 계보)을 물려받아, '확률 엔진'만 신경망으로 바꿉니다.
# 이름은 신경망 시대에 맞춰 NeuralLM 으로, 아래에서 NGramLM 으로도 노출(web_service 호환).
# stoi = string to integer (토큰→인덱스),  itos = integer to string (인덱스→토큰)
class NeuralLM(_load_prev("v0.0.9")):
    # 하이퍼파라미터(HIDDEN / LR / EPOCHS / SEED)는 3.train/train.py 에서 설정해요.

    def __init__(self):
        super().__init__()
        self.stoi = None       # 토큰 -> 정수 인덱스
        self.itos = None       # 정수 인덱스 -> 토큰 (어휘 목록)
        self.net = None        # 학습된 신경망(nn.Module)

    # ---------- 어휘 / 데이터 준비 ----------
    def build_vocab(self, sentences):
        """학습 문장의 모든 토큰(어휘)을 처음 나온 순서대로 모아요. (<END> 포함, 위치=인덱스)"""
        vocab, seen = [], set()
        for s in sentences:
            for tok in self.prepare(self.tokenize(s)):
                if tok not in seen:
                    seen.add(tok)
                    vocab.append(tok)
        return vocab

    def make_pairs(self, sentences):
        """
        문장들을 (앞 토큰 인덱스, 다음 토큰 인덱스) 짝으로 바꿔요. 예) ids=[0,1,2,3,4,5] →
          xs=[0,1,2,3,4] (앞/입력),  ys=[1,2,3,4,5] (다음/정답). ys 의 끝은 항상 <END>.
        """
        xs, ys = [], []
        for s in sentences:
            ids = [self.stoi[t] for t in self.prepare(self.tokenize(s))]
            for a, b in zip(ids, ids[1:]):
                xs.append(a)
                ys.append(b)
        return xs, ys

    # ---------- 신경망 만들기 (하위 버전이 override 해 구조를 바꿀 수 있어요) ----------
    def build_net(self, vocab_size, hidden):
        """이 버전의 신경망(nn.Module)을 만들어요. v0.1.4 는 2토큰용으로 override."""
        return BigramModel(vocab_size, hidden)

    # ---------- 학습 (개수 세기 대신 '경사하강'; 전체 배치 · 수동 갱신) ----------
    def train(self, sentences):
        _require_torch()
        # 1) 어휘 + (앞, 다음) 짝
        self.itos = self.build_vocab(sentences)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        V = len(self.itos)

        xs_list, ys_list = self.make_pairs(sentences)
        xs = torch.tensor(xs_list, dtype=torch.long)
        ys = torch.tensor(ys_list, dtype=torch.long)

        # 2) 신경망(2층) 만들기
        torch.manual_seed(self.SEED)              # 초기화 재현 가능하게
        self.net = self.build_net(V, self.HIDDEN)

        print(f"  학습 시작: 어휘 {V}개, 은닉 {self.HIDDEN}, 짝 {len(xs_list)}개, "
              f"epochs {self.EPOCHS}, lr {self.LR}")
        for epoch in range(1, self.EPOCHS + 1):
            logits = self.net(xs)                 # 순전파 (N, V)
            loss = F.cross_entropy(logits, ys)    # softmax + NLL 한 번에

            for p in self.net.parameters():
                p.grad = None
            loss.backward()                       # autograd 가 기울기 계산
            with torch.no_grad():
                for p in self.net.parameters():
                    p -= self.LR * p.grad         # 경사하강 1스텝 (수동; 옵티마이저는 v0.1.2)

            if epoch == 1 or epoch % 20 == 0:
                print(f"  epoch {epoch:4d}/{self.EPOCHS}   loss {loss.item():.4f}")

        self.net.eval()
        return self.net

    # ---------- 저장 / 불러오기 (PyTorch 표준 · 어휘는 vocab.json 에 따로) ----------
    def save(self, model_path, vocab_path):
        """가중치는 state_dict(model.pt), 어휘는 vocab.json 에 저장."""
        _require_torch()
        torch.save(self.net.state_dict(), model_path)                  # PyTorch 표준 저장
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump({"tokenizer": self.tokenizer_name(), "vocab": self.itos},
                      f, ensure_ascii=False)

    def load(self, model_path):
        """model.pt(가중치) + 같은 폴더의 vocab.json(어휘)로 복원."""
        _require_torch()
        vocab_path = os.path.join(os.path.dirname(model_path), "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.itos = meta["vocab"]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        V = len(self.itos)

        state = torch.load(model_path, map_location="cpu")
        hidden = state["fc1.weight"].shape[0]       # 저장된 가중치에서 은닉 크기 복원
        self.net = self.build_net(V, hidden)
        self.net.load_state_dict(state)
        self.net.eval()
        return self

    def run_train(self, data_path, model_path, vocab_path):
        print(f"데이터를 읽는 중... ({data_path})")
        sentences = self.read_sentences(data_path)
        print(f"문장 {len(sentences)}개 / 토크나이저={self.tokenizer_name()}")
        self.train(sentences)
        self.save(model_path, vocab_path)
        print(f"모델 저장 완료 -> {model_path}  (+ vocab.json)")

    # ---------- 확률 엔진: 개수 표 대신 신경망 forward ----------
    def _context_tensor(self, recent):
        """recent 에서 신경망 입력 텐서를 만들어요. 앞 토큰이 없거나 어휘 밖이면 None."""
        if not recent or recent[-1] not in self.stoi:
            return None
        return torch.tensor([self.stoi[recent[-1]]], dtype=torch.long)   # (1,)

    def _probs(self, recent):
        """다음 토큰 확률 분포 = softmax(net(문맥))."""
        x = self._context_tensor(recent)
        if x is None:
            return None
        with torch.no_grad():
            logits = self.net(x)[0]              # (V,)
        return torch.softmax(logits, dim=0)

    def next_dist(self, recent):
        """
        후보 분포(v0.0.9 accuracy() 가 쓰는 것)를 신경망 버전으로. 개수 표 대신 softmax 라
        **어휘 전체**에 확률이 퍼져 있어요 — 카운트처럼 '표에 없어서 못 맞히는' 일이 없습니다.
        """
        probs = self._probs(recent)
        if probs is None:
            return None
        return {self.itos[j]: float(probs[j]) for j in range(len(self.itos))}

    def token_prob(self, recent, token):
        """퍼플렉서티(v0.0.9)가 부르는 함수. '개수 비율' 대신 '신경망이 준 확률'로 답합니다."""
        if token not in self.stoi:
            return self.FLOOR
        probs = self._probs(recent)
        if probs is None:
            return self.FLOOR
        # softmax 가 아주 작은 확률을 float32 에서 0.0 으로 언더플로할 수 있어요(2층이라 logit↑).
        # 카운트 모델과 '같은 바닥값(FLOOR)'으로 눌러 perplexity 의 log(0) 을 막고 비교도 공정하게.
        prob = float(probs[self.stoi[token]])
        return prob if prob > self.FLOOR else self.FLOOR

    def next_token(self, recent, temperature):
        """다음 토큰을 신경망 확률 분포에서 뽑아요 (온도·top-k·top-p 는 choose() 재사용)."""
        probs = self._probs(recent)
        if probs is None:
            return None
        candidates = {self.itos[j]: float(probs[j]) for j in range(len(self.itos))}
        return self.choose(candidates, temperature)

    def can_continue(self, recent):
        return bool(recent) and recent[-1] in self.stoi

    def generate(self, start_text, temperature=0.0, top_k=0, top_p=1.0):
        """생성. 신경망은 표가 없으므로 여기서 새로 정의(개수 세기 버전과 달리)."""
        self._top_k, self._top_p = top_k, top_p
        recent = self.tokenize(start_text)
        if not recent:
            recent = [self.BOT]        # 시작 토큰이 없으면 '봇 차례'부터
        for _ in range(self.MAX_LENGTH):
            nxt = self.next_token(recent, temperature)
            if nxt is None or self.is_end(nxt):
                break
            recent.append(nxt)
        return self.detokenize(recent)


# web_service(load_version_class) 와 상속 사슬(_load_prev)이 module.NGramLM 을 찾으므로 노출.
NGramLM = NeuralLM


class Model(NeuralLM):
    pass


DATA_PATH = os.path.join(_VERSION_DIR, "1.data", "data.txt")      # 학습용
VALID_PATH = os.path.join(_VERSION_DIR, "1.data", "valid.txt")    # 검증용 (처음 보는 대화)
MODEL_PATH = os.path.join(_HERE, "model.pt")                      # 가중치 (PyTorch 표준)
VOCAB_PATH = os.path.join(_HERE, "vocab.json")                    # 어휘 (토크나이저 + 목록)
