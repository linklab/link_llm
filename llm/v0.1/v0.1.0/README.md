# link_llm — 나만의 작은 언어 모델 (v0.1.0)

**신경망 시대의 첫 버전.** 지금까지는 단어(토큰)를 **세어서** 확률을 만들었다면,
v0.1.0 은 그 확률을 **학습**합니다. (PyTorch autograd 사용)

## v0.0.9 → v0.1.0, 무엇이 바뀌었나요?

> **개수 세기 → 경사하강 학습.** "다음 토큰 확률"을 표에서 세는 대신, 가중치 `W` 를 두고
> 데이터를 잘 맞히도록 조금씩 고쳐요. 미분은 PyTorch **autograd** 가 대신 계산합니다.

### 딱 한 가지만 바뀌었어요 — '확률 엔진'

토크나이저 · `<END>` · 대화(`chat`) · 온도/top-k·top-p 샘플링 · 퍼플렉서티(PPL) 는
**v0.0.x 것을 그대로 물려받아요.** 바뀐 건 오직:

| | v0.0.9 (개수 세기) | v0.1.0 (신경망) |
|---|---|---|
| 다음 토큰 확률 | 표에서 `count / total` | `softmax(W[앞 토큰])` |
| 학습 | 개수를 셈 | `W` 를 경사하강으로 학습 |
| 미분 | 없음 | PyTorch `loss.backward()` |

### 신경망 모델 (nn.Module) + 한 스텝의 학습

강의 자료 **06.fcn** 의 `nn.Module` 스타일로, 앞 토큰(one-hot) → `nn.Linear` → 다음 토큰 점수(logits):

```python
import torch.nn as nn

class BigramModel(nn.Module):                  # 06.fcn 의 MyFirstModel 처럼
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(vocab_size, vocab_size, bias=False)   # W: V -> V
    def forward(self, x):                      # x: 앞 토큰 인덱스 (B,)
        onehot = F.one_hot(x, num_classes=self.vocab_size).float()
        return self.linear(onehot)             # logits (B, V)

model  = BigramModel(V)
logits = model(xs)                      # 순전파 (forward)
loss   = F.cross_entropy(logits, ys)    # softmax + NLL 을 한 번에
for p in model.parameters(): p.grad = None
loss.backward()                         # autograd 가 기울기 계산
with torch.no_grad():
    for p in model.parameters(): p -= LR * p.grad   # 경사하강 1스텝 (수동 갱신)
```

> **핵심 통찰:** 학습이 끝난 신경망 bigram 은 **개수 bigram 과 사실상 같아져요.**
> 경사하강이 결국 "그 문맥에서의 다음 토큰 등장 비율"을 재현하거든요.

## 실행 방법

> ⚠️ 이 버전부터 **PyTorch** 가 필요해요. torch 를 설치할 수 있는 파이썬 환경에서 실행하세요.
> ```bash
> pip install torch
> ```

```bash
python3 3.train/train.py     # 학습 → 2.models/model.pt + 3.train/loss.svg (손실 곡선)
python3 4.test/test.py       # 평가 (학습/검증 PPL + 대화 예시)
```

학습으로 `model.json` 이 생기면, 루트의 웹앱(`web_service`)에서도 v0.1.0 을 골라
**대화로 직접 평가**할 수 있어요. (모든 버전이 같은 구조라 웹앱이 수정 없이 로드해요.)

## 완결성 — 웹앱에서 바로 평가

이 버전은 **학습 → 생성/대화 → PPL 측정**까지 한 버전에 완결돼요.
`1.data / 2.models / 3.train / 4.test` 구조도 v0.0.x 와 똑같아서 `web_service` 가 그대로 불러옵니다.

## 여기서부터 신경망 (v0.1.x, 5단계)

- **v0.1.0 신경망 bigram + autograd ← 현재** (첫 신경망 모델, 완결)
- v0.1.1 미니배치·벡터화 학습 → v0.1.2 옵티마이저(`torch.optim`) → v0.1.3 정규화·초기화
- v0.1.4 **기준선 대결(캡스톤)** — 웹앱에서 v0.0.9(카운트) vs 신경망 bigram 을 PPL 로 나란히

> 다음 v0.2.x 에서는 one-hot 을 **임베딩 벡터**로 바꾸고 은닉층을 얹어 **MLP(Bengio)** 로 갑니다.

## 손실 곡선 — `3.train/loss.svg`

학습이 끝나면 **에폭별 손실**을 SVG 그래프로 남겨요. 숫자만 흘려보내지 말고
"정말 내려가고 있나"를 눈으로 확인하는 게 신경망 학습의 첫 습관이에요.

- matplotlib 을 쓰지 않아요. 선 하나 그리는 데 필요한 건 좌표 계산이 전부라
  `save_loss_plot()` 이 SVG 를 직접 씁니다 (의존성은 여전히 torch 하나).
- 이 버전은 **full-batch 수동 SGD** 라 곡선이 완만하고, 1,000 에폭에서도
  **아직 내려가는 중**이에요 — 덜 수렴했다는 뜻이고, v0.1.1(미니배치)·v0.1.2(Adam)에서
  같은 에폭 수로 훨씬 낮은 지점에 도달하는 걸 곡선끼리 비교하면 바로 보입니다.
- `save_loss_plot()` 은 v0.1.0 에 있어서 **이후 모든 신경망 버전이 그대로 물려받아요.**
