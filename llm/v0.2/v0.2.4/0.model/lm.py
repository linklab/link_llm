# -*- coding: utf-8 -*-
"""
lm.py  (v0.2.4)  -  임베딩 시각화 (PCA · t-SNE · 최근접이웃)

[이 버전의 성격 — 모델은 그대로]
  지금까지는 버전마다 '모델'이나 '학습'을 바꿨어요. v0.2.4 는 **아무것도 바꾸지 않습니다.**
  구조·학습·저장은 v0.2.3 그대로이고, **학습이 끝난 임베딩 표를 들여다보는 도구**만 더해요.
  → "임베딩이 정말 비슷한 토큰을 비슷한 벡터로 배치했나?" 를 눈과 숫자로 확인하는 버전.

[더하는 도구 3개]
  ① **최근접이웃**  : 코사인 유사도로 "이 토큰과 가장 가까운 토큰들" 을 찾아요.
                      임베딩이 의미를 담았는지 **가장 직접적으로** 보여줍니다.
  ② **PCA**         : E차원 임베딩을 **2차원으로 눌러** 산점도로. 분산을 가장 많이 남기는
                      두 방향(주성분)을 찾는 선형 방법 — `torch.linalg.svd` 한 번이면 끝나요.
  ③ **t-SNE**       : 이웃 관계를 보존하는 비선형 방법. 고차원 유사도 P 와 저차원 유사도 Q 의
                      KL(P‖Q) 를 **autograd 로 최소화**해 구현했어요 (v0.1.0 에서 배운 그 autograd).

[의존성]  여전히 **torch 하나**예요. numpy·matplotlib·scikit-learn 을 쓰지 않고,
          SVD 는 torch, 산점도는 SVG 를 직접 씁니다 (v0.1.0 의 loss.svg 와 같은 방식).

[그대로인 것]  build_net · train · save · load · 대화 · PPL 전부 v0.2.3 상속. 새 함수만 추가.

[주의]  PyTorch 필요.   pip install torch
"""

import os
import math
import importlib.util

try:
    import torch
except ImportError:
    torch = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERSION_DIR = os.path.dirname(_HERE)


def _load_prev_module(prev_version):
    """이전 버전 lm.py 모듈을 통째로 불러와요 (NGramLM 계보 재사용)."""
    group = prev_version.rsplit(".", 1)[0]                    # "v0.2.3" -> "v0.2"
    llm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../llm 루트
    path = os.path.join(llm_dir, group, prev_version, "0.model", "lm.py")
    spec = importlib.util.spec_from_file_location("llmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_torch():
    if torch is None:
        raise SystemExit("이 버전(v0.2.4)은 PyTorch 가 필요해요.\n  pip install torch")


# v0.2.3(일반화·튜닝)을 물려받아, 임베딩을 들여다보는 도구만 더합니다.
_prev = _load_prev_module("v0.2.3")


class NeuralLM(_prev.NGramLM):
    # 모델·학습 관련 설정은 전부 v0.2.3 것을 그대로 씁니다. 아래는 시각화 설정.
    PLOT_TOKENS = 200      # 산점도에 찍을 토큰 수 (학습 데이터에서 자주 나온 순)
    TSNE_PERPLEXITY = 20.0 # t-SNE 이웃 폭 — 작으면 지역 구조, 크면 전역 구조
    TSNE_STEPS = 500       # t-SNE 경사하강 반복 수
    TSNE_LR = 100.0        # t-SNE 학습률 (보통 10~1000)
    SEED = 1234            # 시각화 재현용 (학습 때는 1.train/train.py 값이 덮어써요)

    # ---------- 임베딩 꺼내기 ----------
    def embedding_matrix(self):
        """학습된 임베딩 표 `emb.weight` (V×E) 를 CPU 텐서로 돌려줘요.
        분석은 CPU 에서 합니다 — 한 번만 하는 계산이고, MPS 는 SVD 를 CPU 로 되돌리거든요."""
        _require_torch()
        return self.net.emb.weight.detach().cpu()

    def token_frequency(self, sentences):
        """학습 문장에서 토큰별 등장 횟수. (산점도에 '자주 쓰는 토큰'만 찍으려고)"""
        freq = {}
        for s in sentences:
            for t in self.prepare(self.tokenize(s)):
                freq[t] = freq.get(t, 0) + 1
        return freq

    def plot_tokens(self, sentences=None, limit=None):
        """찍을 토큰의 인덱스 목록. sentences 를 주면 **자주 나온 순**, 없으면 어휘 순."""
        limit = limit or self.PLOT_TOKENS
        skip = {getattr(self, "PAD", "<PAD>")}
        if sentences:
            freq = self.token_frequency(sentences)
            order = sorted((t for t in self.itos if t not in skip),
                           key=lambda t: -freq.get(t, 0))
        else:
            order = [t for t in self.itos if t not in skip]
        return [self.stoi[t] for t in order[:limit]]

    # ---------- ① 최근접이웃 (코사인 유사도) ----------
    def nearest(self, token, k=5):
        """
        `token` 과 임베딩이 가장 가까운 토큰 k개를 [(토큰, 유사도), ...] 로.

        코사인 유사도 = 두 벡터가 이루는 각도. 길이를 1로 맞춘 뒤 내적하면 나와요.
        1 에 가까울수록 '같은 방향' = 모델이 비슷하게 취급하는 토큰.
        """
        _require_torch()
        if token not in self.stoi:
            return None
        E = self.embedding_matrix()
        E = E / E.norm(dim=1, keepdim=True).clamp(min=1e-12)   # 길이 1로
        sims = E @ E[self.stoi[token]]                          # (V,)
        sims[self.stoi[token]] = -2.0                           # 자기 자신 제외
        pad = getattr(self, "PAD", None)
        if pad in self.stoi:
            sims[self.stoi[pad]] = -2.0                         # <PAD> 제외
        top = torch.topk(sims, k)
        return [(self.itos[i], float(v)) for v, i in zip(top.values, top.indices)]

    # ---------- ② PCA (선형: 분산이 가장 큰 두 방향) ----------
    def pca_2d(self, idx=None):
        """
        임베딩을 2차원으로 투영해요. 반환 = (좌표 (N,2), 설명된 분산 비율).

        원리: 평균을 빼고(중심화) **SVD** 를 하면 특이벡터가 곧 주성분이에요.
        상위 2개 방향에 투영하면 '분산을 가장 많이 남기는' 2차원 그림이 됩니다.
        """
        _require_torch()
        E = self.embedding_matrix()
        if idx is not None:
            E = E[torch.tensor(idx, dtype=torch.long)]
        Ec = E - E.mean(dim=0, keepdim=True)                    # 중심화
        _, S, Vh = torch.linalg.svd(Ec, full_matrices=False)
        coords = Ec @ Vh[:2].T                                  # (N, 2)
        ratio = float((S[:2] ** 2).sum() / (S ** 2).sum().clamp(min=1e-12))
        return coords, ratio

    # ---------- ③ t-SNE (비선형: 이웃 관계 보존) ----------
    def _tsne_affinities(self, E, perplexity):
        """
        고차원 유사도 P 를 만들어요.

        각 점마다 가우시안 폭(beta)을 **이분 탐색**으로 정합니다 — 이웃 수가 대략
        `perplexity` 가 되도록. (조밀한 곳은 좁게, 성긴 곳은 넓게 보라는 뜻)
        """
        n = E.shape[0]
        d2 = torch.cdist(E, E) ** 2                             # 제곱 거리 (n, n)
        P = torch.zeros(n, n)
        target = math.log(perplexity)
        for i in range(n):
            row = torch.cat([d2[i, :i], d2[i, i + 1:]])
            lo, hi, beta = 1e-10, 1e10, 1.0
            for _ in range(50):
                p = torch.exp(-row * beta)
                s = p.sum().clamp(min=1e-12)
                H = torch.log(s) + beta * (row * p).sum() / s   # 엔트로피
                if abs(float(H) - target) < 1e-5:
                    break
                if float(H) > target:                            # 너무 넓다 → beta↑
                    lo = beta
                    beta = beta * 2 if hi == 1e10 else (beta + hi) / 2
                else:                                            # 너무 좁다 → beta↓
                    hi = beta
                    beta = beta / 2 if lo == 1e-10 else (beta + lo) / 2
            p = p / p.sum().clamp(min=1e-12)
            P[i, :i], P[i, i + 1:] = p[:i], p[i:]
        P = (P + P.T) / (2 * n)                                  # 대칭화
        return P.clamp(min=1e-12)

    def tsne_2d(self, idx=None, steps=None, lr=None, perplexity=None, seed=None, log=None):
        """
        t-SNE 로 2차원 좌표를 구해요. 반환 = (좌표 (N,2), 마지막 KL 값).

        저차원 유사도 Q 는 **t-분포**(꼬리가 두꺼움)를 써서, 멀리 있는 점들이
        서로 밀어낼 여지를 줍니다 — 이게 t-SNE 의 't'.
        KL(P‖Q) 를 **autograd 로** 줄여요 (직접 미분 공식을 쓰지 않습니다).
        """
        _require_torch()
        steps = steps or self.TSNE_STEPS
        lr = lr or self.TSNE_LR
        perplexity = perplexity or self.TSNE_PERPLEXITY
        E = self.embedding_matrix()
        if idx is not None:
            E = E[torch.tensor(idx, dtype=torch.long)]
        n = E.shape[0]
        if perplexity >= n / 3:                                  # 점이 적으면 perplexity 를 낮춰요
            perplexity = max(2.0, n / 3 - 1)

        P = self._tsne_affinities(E, perplexity)
        torch.manual_seed(self.SEED if seed is None else seed)
        Y = (torch.randn(n, 2) * 1e-2).requires_grad_(True)
        opt = torch.optim.Adam([Y], lr=lr)
        eye = torch.eye(n, dtype=torch.bool)

        loss = torch.tensor(0.0)
        for step in range(steps):
            # 초반 250스텝은 P 를 부풀려(early exaggeration) 덩어리를 먼저 뭉치게 해요
            Pe = P * 4.0 if step < min(250, steps // 2) else P
            num = 1.0 / (1.0 + torch.cdist(Y, Y) ** 2)           # t-분포 커널
            num = num.masked_fill(eye, 0.0)
            Q = (num / num.sum().clamp(min=1e-12)).clamp(min=1e-12)
            loss = (Pe * (Pe.clamp(min=1e-12).log() - Q.log())).sum()   # KL(P‖Q)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if log and (step == 0 or (step + 1) % 100 == 0):
                log(f"    t-SNE step {step + 1:4d}/{steps}   KL {loss.item():.4f}")
        return Y.detach(), loss.item()

    # ---------- 산점도: 의존성 없이 SVG 를 직접 그려요 ----------
    def save_embedding_plot(self, path, coords, labels, title="", note=""):
        """
        2차원 좌표를 SVG 산점도로 저장해요. (matplotlib 없이 — loss.svg 와 같은 방식)

        토큰 이름을 점 옆에 직접 써서, "무엇이 무엇 옆에 있나" 를 바로 읽을 수 있게 했어요.
        """
        xs = [float(c[0]) for c in coords]
        ys = [float(c[1]) for c in coords]
        if not xs:
            return None

        W, H = 900, 720
        L, R, T, B = 40, 40, 46, 34
        pw, ph = W - L - R, H - T - B

        def span(vals):
            lo, hi = min(vals), max(vals)
            pad = (hi - lo) * 0.06 or 1.0
            return lo - pad, hi + pad

        x0, x1 = span(xs)
        y0, y1 = span(ys)
        px = lambda v: L + pw * (v - x0) / (x1 - x0)
        py = lambda v: T + ph * (y1 - v) / (y1 - y0)          # y 축 뒤집기(위가 큰 값)

        out = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" font-family="sans-serif">',
            f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
            f'<text x="{L}" y="26" font-size="16" fill="#111">{title}</text>',
        ]
        if note:
            out.append(f'<text x="{W - R}" y="26" font-size="11" fill="#666" '
                       f'text-anchor="end">{note}</text>')

        for x, y, name in zip(xs, ys, labels):
            cx, cy = px(x), py(y)
            safe = (name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.6" fill="#2b6cb0" '
                       f'fill-opacity="0.65"/>')
            out.append(f'<text x="{cx + 4:.1f}" y="{cy + 3:.1f}" font-size="9" '
                       f'fill="#333">{safe}</text>')

        out.append(f'<text x="{L}" y="{H - 12}" font-size="11" fill="#666">'
                   f'점 {len(xs)}개 · 이름이 겹치면 확대해서 보세요 (SVG 는 벡터라 깨지지 않아요)</text>')
        out.append("</svg>")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out))
        return path


# web_service / 상속 사슬이 module.NGramLM 을 찾으므로 노출.
NGramLM = NeuralLM


class Model(NeuralLM):
    pass


_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_VERSION_DIR)))  # 저장소 루트
_DATA_DIR = os.path.join(_ROOT, "data")                                  # 모든 버전 공용 데이터
DATA_PATH = os.path.join(_DATA_DIR, "pretrain", "train.txt")      # 학습용
VALID_PATH = os.path.join(_DATA_DIR, "pretrain", "valid.txt")    # 검증용
MODEL_PATH = os.path.join(_HERE, "model.pt")         # 가중치 (PyTorch 표준)
VOCAB_PATH = os.path.join(_HERE, "vocab.json")       # 어휘 (0번=<PAD>)
