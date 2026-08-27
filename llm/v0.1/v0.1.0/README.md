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
python3 1.train/train.py     # 학습 → 0.model/model.pt + 1.train/loss.svg (손실 곡선)
python3 2.test/test.py       # 평가 (학습/검증 PPL + 대화 예시)
```

학습으로 `model.json` 이 생기면, 루트의 웹앱(`web_service`)에서도 v0.1.0 을 골라
**대화로 직접 평가**할 수 있어요. (모든 버전이 같은 구조라 웹앱이 수정 없이 로드해요.)

## 완결성 — 웹앱에서 바로 평가

이 버전은 **학습 → 생성/대화 → PPL 측정**까지 한 버전에 완결돼요.
`0.model / 1.train / 2.test` 구조(+ 루트 공용 `data/`)도 v0.0.x 와 똑같아서 `web_service` 가 그대로 불러옵니다.

## 여기서부터 신경망 (v0.1.x, 5단계)

- **v0.1.0 신경망 bigram + autograd ← 현재** (첫 신경망 모델, 완결)
- v0.1.1 미니배치·벡터화 학습 → v0.1.2 옵티마이저(`torch.optim`) → v0.1.3 정규화·초기화
- v0.1.4 **기준선 대결(캡스톤)** — 웹앱에서 v0.0.9(카운트) vs 신경망 bigram 을 PPL 로 나란히

> 다음 v0.2.x 에서는 one-hot 을 **임베딩 벡터**로 바꾸고 은닉층을 얹어 **MLP(Bengio)** 로 갑니다.

## 손실 곡선 — `1.train/loss.svg`

학습이 끝나면 **에폭별 손실**을 SVG 그래프로 남겨요. 숫자만 흘려보내지 말고
"정말 내려가고 있나"를 눈으로 확인하는 게 신경망 학습의 첫 습관이에요.

- matplotlib 을 쓰지 않아요. 선 하나 그리는 데 필요한 건 좌표 계산이 전부라
  `save_loss_plot()` 이 SVG 를 직접 씁니다 (의존성은 여전히 torch 하나).
- 파란 실선이 **학습 손실**(왼쪽 축), 빨간 점선이 **검증 PPL**(오른쪽 축, 로그 눈금)이에요.
  세로 점선은 조기 종료가 **채택한 에폭**입니다.
- 이 버전은 **full-batch 수동 SGD** 라 곡선이 완만하고, 1,000 에폭에서도
  **아직 내려가는 중**이에요 — 덜 수렴했다는 뜻이고, v0.1.1(미니배치)·v0.1.2(Adam)에서
  같은 에폭 수로 훨씬 낮은 지점에 도달하는 걸 곡선끼리 비교하면 바로 보입니다.
- `save_loss_plot()` 은 v0.1.0 에 있어서 **이후 모든 신경망 버전이 그대로 물려받아요.**
- `loss.svg` 는 model.pt 처럼 **학습 산출물이라 커밋하지 않아요**(`.gitignore`).
  클론 직후에는 없고, `1.train/train.py` 를 한 번 돌리면 생깁니다.

## 조기 종료 (early stopping)

에폭 수를 손으로 맞추는 일을 없앱니다. 매 에폭 **검증 PPL**을 재서

- 좋아지면 → 그때의 가중치를 통째로 기억해 두고,
- `PATIENCE` 에폭 동안 나아지지 않으면 → 멈춘 뒤 **가장 좋았던 가중치로 되돌려** 저장해요.

그래서 `EPOCHS` 는 이제 '정확히 맞춰야 하는 값'이 아니라 넉넉한 **상한**이면 됩니다.
마지막 에폭이 아니라 최고점이 저장되므로, 상한을 크게 잡아도 손해가 없어요.

```python
m.EARLY_STOPPING = True    # False 면 EPOCHS 를 끝까지
m.PATIENCE = 100           # 몇 에폭까지 참을지
m.MIN_DELTA = 0.0          # 이만큼은 좋아져야 '개선'
```

> ⚠️ **PyTorch 본체에는 early stopping 이 없어요.** PyTorch Lightning 의
> `EarlyStopping` 콜백이나 Ignite 의 `EarlyStopping` 핸들러는 **별도 패키지**입니다.
> 이 저장소는 의존성을 torch 하나로 유지하려고, 표준 동작(patience + best 가중치 복원)을
> `lm.py` 안에 15줄짜리 `EarlyStopping` 클래스로 직접 구현했어요.
> `torch.optim.lr_scheduler.ReduceLROnPlateau` 가 쓰는 patience 개념과 같은 방식입니다.

**`PATIENCE` 는 버전마다 달라요** — '한 에폭에 얼마나 배우느냐'가 다르기 때문이에요.

| 버전 | 에폭당 갱신 | PATIENCE | 이유 |
|---|---|---|---|
| v0.1.0 | 1번 (full-batch) | 100 | 아주 느리게 내려가 평탄 구간이 길어요 |
| v0.1.1 | 35번 (미니배치, 수동) | 50 | 여전히 평탄 구간이 김 |
| v0.1.2 ~ v0.2.0 | 35번 (Adam) | 10 | 빨리 수렴해 10이면 충분 |

`PATIENCE` 를 너무 작게 잡으면 **평탄 구간을 수렴으로 오해**해서 일찍 멈춰요.
실제로 v0.1.1 에 patience=10 을 주면 검증 PPL 60.94 에서 멈추지만, 50 을 주면 **38.10** 까지 갑니다.
