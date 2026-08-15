# -*- coding: utf-8 -*-
"""
lm.py  (v0.1.0)  -  신경망 bigram + PyTorch autograd  (드디어 '세기'가 아니라 '학습')

[개수 세기 시대(v0.0.x)와 무엇이 다른가]
  지금까지는 "앞 토큰 다음에 무엇이 몇 번 나왔나"를 **세어서** 확률을 만들었어요.
  v0.1.0 은 그 확률을 **학습**합니다.
    - 가중치 행렬 W (어휘수 V × V) 를 `nn.Module` 로 감싼 아주 작은 신경망(BigramModel)을 두고,
      (강의 자료 06.fcn 스타일: nn.Linear 한 층 + forward)
    - 경사하강법(gradient descent)으로 W 를 조금씩 고쳐 데이터를 잘 맞히게 만들어요.
    - 미분(기울기)은 PyTorch 의 **autograd** (loss.backward()) 가 대신 계산해 줍니다.

[핵심 통찰]  학습이 끝난 신경망 bigram 은 **개수 bigram 과 사실상 같아져요.**
  경사하강이 결국 "그 문맥에서의 다음 토큰 등장 비율"을 재현하기 때문이에요.
  (그래서 v0.1.4 에서 v0.0.9 카운트 모델과 PPL 로 나란히 비교합니다.)

[인터페이스는 그대로 — '확률 엔진'만 교체]
  토크나이저 · <END> · 대화(chat) · 온도/top-k·top-p 샘플링 · 퍼플렉서티(PPL) 는
  v0.0.x 것을 **그대로 물려받고**, 딱 하나
    "다음 토큰 확률을 어디서 얻느냐" (개수 표  ->  신경망 W)
  만 바꿉니다. 그래서 웹앱/평가 코드가 수정 없이 그대로 동작해요.

[주의]  이 버전부터 PyTorch 가 필요합니다.   pip install torch
  (v0.0.x 는 표준 라이브러리만 썼지만, 신경망 시대부터는 torch 를 씁니다.)
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


# ---------- 신경망 bigram 모델 (강의 자료 06.fcn 스타일: nn.Module 상속 + forward) ----------
_Module = nn.Module if torch is not None else object   # torch 없이도 이 파일이 로딩되게


class BigramModel(_Module):
    """
    앞 토큰(one-hot) -> nn.Linear -> 다음 토큰 점수(logits) 를 내는 아주 작은 신경망.
    은닉층 없는 '완전연결망 한 층' 이에요 (06.fcn 의 MyFirstModel 처럼 nn.Module 스타일).
      - __init__ : 레이어(nn.Linear) 를 정의
      - forward  : 입력 x(앞 토큰 인덱스) -> one-hot -> Linear -> logits
    (은닉층·비선형·임베딩은 v0.2.x 에서 더합니다. 여기선 딱 한 층.)
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        # W: 어휘수 V -> V (bias 없음 — 개수 bigram 과 대응되도록)
        self.linear = nn.Linear(vocab_size, vocab_size, bias=False)

    def forward(self, x):                          # x: 앞 토큰 인덱스 (B,)
        onehot = F.one_hot(x, num_classes=self.vocab_size).float()   # (B, V)
        return self.linear(onehot)                 # logits (B, V)


# v0.0.9 의 NGramLM(개수 세기 계보)을 물려받아, '확률 엔진'만 신경망으로 바꿉니다.
# 이름은 신경망 시대에 맞춰 NeuralLM 으로 바꾸되, 아래에서 NGramLM 으로도 노출해
# (web_service 와 상속 사슬이 module.NGramLM 을 찾으므로) 기존 인프라와 호환시켜요.
# stoi = string to integer — 토큰(문자열) → 정수 인덱스 매핑 (dict)
# itos = integer to string — 정수 인덱스 → 토큰(문자열) 매핑 (list/dict)
class NeuralLM(_load_prev("v0.0.9")):
    # 하이퍼파라미터(LR / EPOCHS / SEED)는 3.train/train.py 에서 설정해요.

    def __init__(self):
        super().__init__()
        self.stoi = None       # 토큰 -> 정수 인덱스
        self.itos = None       # 정수 인덱스 -> 토큰 (어휘 목록)
        self.W = None          # 학습된 가중치 (V x V) 텐서

    # ---------- 어휘 / 데이터 준비 ----------
    def build_vocab(self, sentences):
        """
        학습 문장에서 나오는 모든 토큰(어휘)을 '처음 나온 순서대로' 모읍니다. (<END> 포함)

        예) sentences = ["<사용자> 안녕 <봇> 안녕하세요 !"] 이면
          1) tokenize + prepare -> ["<사용자>", "안녕", "<봇>", "안녕하세요", "!", "<END>"]
          2) 처음 보는 토큰만 순서대로 담아 ->
             vocab = ["<사용자>", "안녕", "<봇>", "안녕하세요", "!", "<END>"]
             (이 위치가 곧 토큰의 정수 인덱스: <사용자>=0, 안녕=1, <봇>=2, ... <END>=5)
          여러 문장이면 이미 담긴 토큰은 건너뛰고(중복 제거), 새 토큰만 뒤에 이어 붙여요.
        """
        vocab, seen = [], set()
        for s in sentences:
            for tok in self.prepare(self.tokenize(s)):
                if tok not in seen:
                    seen.add(tok)
                    vocab.append(tok)
        return vocab

    def make_pairs(self, sentences):
        """
        문장들을 (앞 토큰 인덱스, 다음 토큰 인덱스) 짝의 목록으로 바꿉니다.
        이 짝이 곧 신경망 bigram 의 학습 문제 "앞 토큰 -> 다음 토큰" 이에요.

        예) "<사용자> 안녕 <봇> 안녕하세요 !"  (stoi: <사용자>=0, 안녕=1, <봇>=2,
            안녕하세요=3, !=4, <END>=5) 이면
          1) tokenize + prepare -> ["<사용자>", "안녕", "<봇>", "안녕하세요", "!", "<END>"]
          2) stoi 로 정수화        -> ids = [0, 1, 2, 3, 4, 5]
          3) 이웃한 짝으로 자름      -> (0,1) (1,2) (2,3) (3,4) (4,5)
             xs = [0, 1, 2, 3, 4]   # 앞 토큰(입력)
             ys = [1, 2, 3, 4, 5]   # 다음 토큰(정답)
          즉, '<사용자> 다음엔 안녕', '안녕 다음엔 <봇>' … 을 맞히도록 학습해요.
          (문장이 여러 개면 각 문장의 짝들을 xs·ys 뒤에 계속 이어 붙입니다.)
          즉, ys 의 마지막에는 무조건 <END>가 있음.
        """
        xs, ys = [], []
        for s in sentences:
            ids = [self.stoi[t] for t in self.prepare(self.tokenize(s))]
            for a, b in zip(ids, ids[1:]):
                xs.append(a)
                ys.append(b)
        return xs, ys

    # ---------- 학습 (개수 세기 대신 '경사하강'; 모델은 nn.Module) ----------
    def train(self, sentences):
        _require_torch()
        # 1) 어휘 사전
        self.itos = self.build_vocab(sentences)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        V = len(self.itos)

        # 2) (앞, 다음) 짝을 텐서로
        xs_list, ys_list = self.make_pairs(sentences)
        xs = torch.tensor(xs_list, dtype=torch.long)
        ys = torch.tensor(ys_list, dtype=torch.long)

        # 3) 모델 만들기 (nn.Module). 앞으로 model(x) 로 순전파해요.
        torch.manual_seed(self.SEED)              # 초기화 재현 가능하게
        model = BigramModel(V)

        print(f"  학습 시작: 어휘 {V}개, 짝 {len(xs_list)}개, epochs {self.EPOCHS}, lr {self.LR}")
        for epoch in range(1, self.EPOCHS + 1):
            # --- 순전파: model(x) 가 forward() 를 불러 logits 를 냄 ---
            logits = model(xs)                    # (N, V)
            loss = F.cross_entropy(logits, ys)    # softmax + NLL 을 한 번에 (수치 안정)

            # --- 역전파: autograd 가 각 파라미터의 기울기(.grad)를 계산 ---
            for p in model.parameters():
                p.grad = None
            loss.backward()

            # --- 경사하강 1스텝 (수동 갱신; 옵티마이저는 v0.1.2에서) ---
            with torch.no_grad():
                for p in model.parameters():
                    p -= self.LR * p.grad

            if epoch == 1 or epoch % 20 == 0:
                print(f"  epoch {epoch:4d}/{self.EPOCHS}   loss {loss.item():.4f}")

        # 추론/저장이 쓰는 self.W 로 보관. (self.W[prev] = 그 앞 토큰 다음의 logits)
        #   nn.Linear.weight 는 (out, in) 이라 one-hot 입력에선 logits[j]=weight[j, prev].
        #   '앞 토큰 -> logits' 로 바로 인덱싱하려고 전치(.t())해 self.W[prev] 에 맞춰요.
        self.W = model.linear.weight.detach().t().contiguous()
        return self.W

    # ---------- 저장 / 불러오기 ----------
    def to_dict(self, W):
        return {
            "type": "neural_bigram",
            "tokenizer": self.tokenizer_name(),
            "vocab": self.itos,          # itos (인덱스=위치)
            "W": W.tolist(),             # 학습된 가중치 (V x V)
        }

    def save(self, W, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(W), f, ensure_ascii=False)

    def load(self, path):
        _require_torch()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.model = data
        self.itos = data["vocab"]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.W = torch.tensor(data["W"], dtype=torch.float32)
        return self

    def run_train(self, data_path, model_path):
        print(f"데이터를 읽는 중... ({data_path})")
        sentences = self.read_sentences(data_path)
        print(f"문장 {len(sentences)}개 / 토크나이저={self.tokenizer_name()}")
        W = self.train(sentences)
        print(f"학습 완료: 가중치 {tuple(W.shape)}")
        self.save(W, model_path)
        print(f"모델 저장 완료 -> {model_path}")

    # ---------- 확률 엔진: 개수 표 대신 신경망 W ----------
    def _probs_for(self, prev_token):
        """앞 토큰(prev) 다음에 올 토큰들의 확률 분포 = softmax(W[prev])."""
        logits = self.W[self.stoi[prev_token]]
        return torch.softmax(logits, dim=0)

    def token_prob(self, recent, token):
        """
        퍼플렉서티(v0.0.9)가 부르는 함수. '개수 비율' 대신 '신경망이 준 확률'로 답합니다.
        (softmax 라 모든 토큰에 0보다 큰 확률이 있어요. 어휘에 없는 토큰만 바닥값 FLOOR.)
        """
        if not recent:
            return self.FLOOR
        prev = recent[-1]
        if prev not in self.stoi or token not in self.stoi:
            return self.FLOOR
        return float(self._probs_for(prev)[self.stoi[token]])

    def next_token(self, recent, temperature):
        """
        다음 토큰을 '신경망 확률 분포'에서 뽑습니다.
        분포만 신경망으로 만들고, 온도·top-k·top-p 샘플링은 v0.0.7 의 choose() 를 그대로 재사용.
        """
        if not recent:
            return None
        prev = recent[-1]
        if prev not in self.stoi:
            return None
        probs = self._probs_for(prev)
        # choose() 가 쓰는 {토큰: 점수} 형태로. (여기선 점수 = 확률)
        candidates = {self.itos[j]: float(probs[j]) for j in range(len(self.itos))}
        return self.choose(candidates, temperature)

    def can_continue(self, recent):
        return bool(recent) and recent[-1] in self.stoi

    def generate(self, start_text, temperature=0.0, top_k=0, top_p=1.0):
        """
        생성. (개수 세기 버전의 generate 는 표에서 무작위 시작점을 골랐지만,
         신경망은 표가 없으므로 여기서 새로 정의해요.)
        """
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
MODEL_PATH = os.path.join(_HERE, "model.json")
