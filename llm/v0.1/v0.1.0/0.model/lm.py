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
import math
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
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_VERSION_DIR)))  # 저장소 루트
_DATA_DIR = os.path.join(_ROOT, "data")                                  # 모든 버전 공용 데이터


def _load_prev(prev_version):
    group = prev_version.rsplit(".", 1)[0]                    # "v0.0.9" -> "v0.0" (마이너 그룹)
    llm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../llm 루트
    path = os.path.join(llm_dir, group, prev_version, "0.model", "lm.py")
    spec = importlib.util.spec_from_file_location("llmlm_" + prev_version.replace(".", "_"), path)
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
# ---------- 조기 종료 (early stopping) ----------
# PyTorch 본체에는 early stopping 이 없어요 (Lightning·Ignite 같은 별도 패키지에 있어요).
# 이 저장소는 의존성을 torch 하나로 유지하려고, 표준 동작을 그대로 15줄로 직접 구현합니다.
#   · 검증 점수가 좋아지면 그때의 가중치를 통째로 기억해 두고,
#   · patience 에폭 동안 나아지지 않으면 멈춘 뒤 **가장 좋았던 가중치로 되돌려요.**
# (torch.optim.lr_scheduler.ReduceLROnPlateau 가 쓰는 patience 개념과 같아요.)
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience        # 몇 에폭까지 참을지
        self.min_delta = min_delta      # 이만큼은 좋아져야 '개선'으로 인정
        self.best = float("inf")        # 지금까지 가장 좋은(=낮은) 검증 점수
        self.best_epoch = 0
        self.best_state = None          # 그때의 가중치 사본
        self.bad_epochs = 0

    def step(self, score, epoch, model):
        """이번 에폭 점수를 넣어요. 멈춰야 하면 True."""
        if score < self.best - self.min_delta:
            self.best, self.best_epoch, self.bad_epochs = score, epoch, 0
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model):
        """가장 좋았던 가중치로 되돌려요 (마지막 에폭이 아니라 최고점을 저장하려고)."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# 이름은 신경망 시대에 맞춰 NeuralLM 으로, 아래에서 NGramLM 으로도 노출(web_service 호환).
# stoi = string to integer (토큰→인덱스),  itos = integer to string (인덱스→토큰)
class NeuralLM(_load_prev("v0.0.9")):
    # 하이퍼파라미터(HIDDEN / LR / EPOCHS / SEED)는 1.train/train.py 에서 설정해요.

    # 조기 종료 기본값 (1.train/train.py 에서 바꿔요)
    EARLY_STOPPING = True      # False 면 EPOCHS 를 끝까지 돕니다
    PATIENCE = 10              # 검증 PPL 이 이만큼 안 좋아지면 중단
    MIN_DELTA = 0.0            # 개선으로 인정할 최소 폭

    def __init__(self):
        super().__init__()
        self.stoi = None       # 토큰 -> 정수 인덱스
        self.itos = None       # 정수 인덱스 -> 토큰 (어휘 목록)
        self.net = None        # 학습된 신경망(nn.Module)
        self.losses = []       # 에폭별 학습 손실
        self.valid_scores = [] # 에폭별 검증 PPL (조기 종료 판단 근거)
        self.stopped_epoch = 0 # 실제로 채택된(=가장 좋았던) 에폭

    # ---------- 조기 종료: 하위 버전 train() 들이 공통으로 쓰는 3개 도우미 ----------
    def start_early_stopping(self, valid_sentences):
        """검증 문장이 있고 EARLY_STOPPING 이 켜져 있으면 감시자를 만들어요."""
        self.valid_scores = []
        self._valid_sentences = valid_sentences
        if not valid_sentences or not self.EARLY_STOPPING:
            return None
        print(f"  조기 종료 켬: 검증 PPL 기준, patience={self.PATIENCE}, min_delta={self.MIN_DELTA}")
        return EarlyStopping(self.PATIENCE, self.MIN_DELTA)

    def should_stop_early(self, stopper, epoch):
        """에폭 끝에서 검증 PPL 을 재고 '멈출 때인가'를 알려줘요."""
        if stopper is None:
            return False
        self.net.eval()                                   # 평가 모드로 재고
        score = self.perplexity(self._valid_sentences)    # v0.0.9 의 PPL 을 그대로 사용
        self.net.train()                                  # 다시 학습 모드
        self.valid_scores.append(score)
        if stopper.step(score, epoch, self.net):
            print(f"  ⏹ 조기 종료: {self.PATIENCE}에폭 동안 검증 PPL 개선 없음 "
                  f"(epoch {epoch} 에서 중단)")
            return True
        return False

    def finish_early_stopping(self, stopper):
        """가장 좋았던 가중치로 되돌리고, 그 에폭을 기록해요."""
        if stopper is None:
            return
        stopper.restore(self.net)
        self.stopped_epoch = stopper.best_epoch
        print(f"  ✔ 최고 검증 PPL {stopper.best:.2f} (epoch {stopper.best_epoch}) "
              f"가중치로 되돌려 저장합니다")

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
    def train(self, sentences, valid_sentences=None):
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

        self.losses = []   # 에폭별 손실 (1.train/loss.svg 곡선용)
        stopper = self.start_early_stopping(valid_sentences)
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

            self.losses.append(loss.item())
            if epoch == 1 or epoch % 20 == 0:
                print(f"  epoch {epoch:4d}/{self.EPOCHS}   loss {loss.item():.4f}")
            if self.should_stop_early(stopper, epoch):
                break

        self.finish_early_stopping(stopper)
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

    # ---------- 손실 곡선: 의존성 없이 SVG 를 직접 그려요 ----------
    def save_loss_plot(self, path, title=""):
        """
        에폭별 **학습 손실**(파랑, 왼쪽 축)과 **검증 PPL**(빨강, 오른쪽 축)을 SVG 로 저장합니다.
        조기 종료가 켜져 있으면 채택된 에폭에 세로 점선을 그어요 —
        "학습 손실은 계속 내려가는데 검증은 돌아선다"가 한눈에 보이는 그림이에요.

        matplotlib 을 쓰지 않아요. 선 긋는 데 필요한 건 좌표 계산이 전부라 직접 그립니다.
        (SVG 는 그냥 텍스트라 브라우저·GitHub 에서 바로 보여요.)
        """
        losses = getattr(self, "losses", None)
        if not losses:
            return None                       # 손실을 기록하지 않는 모델(카운트 등)
        valids = getattr(self, "valid_scores", None) or []
        best_epoch = getattr(self, "stopped_epoch", 0)

        W, H = 760, 380                       # 전체 크기
        L, R, T, B = 74, 62, 46, 54           # 여백(왼/오른/위/아래)
        pw, ph = W - L - R, H - T - B         # 그래프 영역
        n = len(losses)

        def span(vals):                       # 값 범위 (평평하면 살짝 벌려요)
            lo, hi = min(vals), max(vals)
            return (lo - 0.5, hi + 0.5) if hi - lo < 1e-9 else (lo, hi)

        lo, hi = span(losses)

        def px(i):                            # 에폭 i(0부터) -> x 좌표
            return L + (pw * i / (n - 1) if n > 1 else pw / 2)

        def py(v):                            # 학습 손실 -> y 좌표
            return T + ph * (hi - v) / (hi - lo)

        out = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" font-family="sans-serif">',
            f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
            f'<text x="{L}" y="26" font-size="16" fill="#111">{title}</text>',
        ]

        # 가로 눈금선 + 왼쪽 축 라벨 (학습 손실)
        for k in range(5):
            v = lo + (hi - lo) * k / 4
            y = py(v)
            out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" '
                       f'stroke="#e5e5e5" stroke-width="1"/>')
            out.append(f'<text x="{L - 10}" y="{y + 4:.1f}" font-size="11" fill="#2b6cb0" '
                       f'text-anchor="end">{v:.2f}</text>')

        # x축 라벨 (에폭)
        for k in range(5):
            i = round((n - 1) * k / 4)
            out.append(f'<text x="{px(i):.1f}" y="{T + ph + 20}" font-size="11" fill="#666" '
                       f'text-anchor="middle">{i + 1}</text>')

        # 축
        out.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T + ph}" stroke="#999"/>')
        out.append(f'<line x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}" stroke="#999"/>')
        out.append(f'<text x="{L + pw / 2}" y="{H - 14}" font-size="12" fill="#444" '
                   f'text-anchor="middle">epoch</text>')
        out.append(f'<text x="18" y="{T + ph / 2}" font-size="12" fill="#2b6cb0" '
                   f'text-anchor="middle" transform="rotate(-90 18 {T + ph / 2})">train loss</text>')

        # 검증 PPL (오른쪽 축). PPL 은 exp(손실) 이라 **로그 눈금**으로 그려요 —
        # 1에폭째 PPL 이 수백이라 선형 눈금이면 바닥의 32~40 구간이 뭉개져 안 보입니다.
        if valids:
            vlo, vhi = min(valids), max(valids)
            if vhi / max(vlo, 1e-9) < 1.05:
                vlo, vhi = vlo * 0.95, vhi * 1.05
            llo, lhi = math.log(vlo), math.log(vhi)

            def vy(v):
                return T + ph * (lhi - math.log(v)) / (lhi - llo)

            vpts = " ".join(f"{px(i):.1f},{vy(v):.1f}" for i, v in enumerate(valids))
            out.append(f'<polyline points="{vpts}" fill="none" stroke="#c53030" '
                       f'stroke-width="1.6" stroke-dasharray="4 2"/>')
            for k in range(5):
                v = math.exp(llo + (lhi - llo) * k / 4)
                out.append(f'<text x="{L + pw + 10}" y="{vy(v) + 4:.1f}" font-size="11" '
                           f'fill="#c53030">{v:.1f}</text>')
            out.append(f'<text x="{W - 14}" y="{T + ph / 2}" font-size="12" fill="#c53030" '
                       f'text-anchor="middle" '
                       f'transform="rotate(90 {W - 14} {T + ph / 2})">valid PPL (log)</text>')

        # 학습 손실 곡선
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(losses))
        out.append(f'<polyline points="{pts}" fill="none" stroke="#2b6cb0" stroke-width="1.8"/>')

        # 조기 종료로 채택된 에폭에 세로 점선
        if best_epoch and 1 <= best_epoch <= n:
            bx = px(best_epoch - 1)
            out.append(f'<line x1="{bx:.1f}" y1="{T}" x2="{bx:.1f}" y2="{T + ph}" '
                       f'stroke="#666" stroke-width="1.2" stroke-dasharray="5 4"/>')
            anchor = "end" if bx > L + pw * 0.6 else "start"
            dx = -6 if anchor == "end" else 6
            out.append(f'<text x="{bx + dx:.1f}" y="{T + 14}" font-size="11" fill="#444" '
                       f'text-anchor="{anchor}">채택 epoch {best_epoch}</text>')

        # 요약
        summary = f'{n} epochs · 학습손실 {losses[0]:.4f} → {losses[-1]:.4f}'
        if valids:
            summary += f' · 검증PPL 최저 {min(valids):.2f}'
        out.append(f'<text x="{L + pw}" y="26" font-size="11" fill="#666" text-anchor="end">'
                   f'{summary}</text>')
        out.append("</svg>")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out))
        return path

    def run_train(self, data_path, model_path, vocab_path, valid_path=None):
        print(f"데이터를 읽는 중... ({data_path})")
        sentences = self.read_sentences(data_path)
        print(f"문장 {len(sentences)}개 / 토크나이저={self.tokenizer_name()}")

        # 조기 종료용 검증 문장 (없으면 EPOCHS 를 끝까지 돕니다)
        valid_sentences = None
        if valid_path and os.path.exists(valid_path):
            valid_sentences = self.read_sentences(valid_path)
            print(f"검증 문장 {len(valid_sentences)}개 ({valid_path})")
        self.train(sentences, valid_sentences)
        self.save(model_path, vocab_path)
        print(f"모델 저장 완료 -> {model_path}  (+ vocab.json)")

        # 손실 곡선은 그 버전의 1.train/ 폴더에 (model_path 기준으로 경로를 되짚어요)
        version_dir = os.path.dirname(os.path.dirname(model_path))
        version = os.path.basename(version_dir)
        plot_path = os.path.join(version_dir, "1.train", "loss.svg")
        if self.save_loss_plot(plot_path, f"{version} 학습 손실 곡선"):
            print(f"손실 곡선 저장 완료 -> {plot_path}")

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


DATA_PATH = os.path.join(_DATA_DIR, "data.txt")      # 학습용
VALID_PATH = os.path.join(_DATA_DIR, "valid.txt")    # 검증용 (처음 보는 대화)
MODEL_PATH = os.path.join(_HERE, "model.pt")                      # 가중치 (PyTorch 표준)
VOCAB_PATH = os.path.join(_HERE, "vocab.json")                    # 어휘 (토크나이저 + 목록)
