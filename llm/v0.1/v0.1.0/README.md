# link_llm — 나만의 작은 언어 모델 (v0.1.0)

**신경망 시대의 첫 버전.** 토큰을 **세어서** 확률을 만들던 것을, 확률을 **학습**하는 방식으로 바꿔요.
(PyTorch autograd 사용)

## 한 걸음 — '확률 엔진'만 교체

바뀐 건 "다음 토큰 확률을 어디서 얻느냐" 하나뿐이에요.

| | v0.0.9 (개수 세기) | v0.1.0 (신경망) |
|---|---|---|
| 다음 토큰 확률 | 표에서 `count / total` | `softmax(net(앞 토큰))` |
| 학습 | 개수를 셈 | 가중치를 경사하강으로 학습 |
| 미분 | 없음 | PyTorch `loss.backward()` |

그대로 물려받는 것: 토크나이저 · `<END>` · 대화(`chat`) · 온도/top-k·top-p 샘플링 · 퍼플렉서티(PPL).

## 모델 — 2층 MLP

- 입력: 앞 토큰 one-hot (V차원)
- 흐름: `fc1(V→H)` → `tanh` → `fc2(H→V)` → logits
- 하이퍼파라미터 `HIDDEN`(H)은 `1.train/train.py` 에서 설정

```python
class BigramModel(nn.Module):
    def __init__(self, vocab_size, hidden):
        super().__init__()
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(vocab_size, hidden)    # 1층: one-hot(V) -> 은닉 H
        self.fc2 = nn.Linear(hidden, vocab_size)    # 2층: 은닉 H -> logits V
    def forward(self, x):                           # x: 앞 토큰 인덱스 (B,)
        onehot = F.one_hot(x, num_classes=self.vocab_size).float()
        hidden = torch.tanh(self.fc1(onehot))
        return self.fc2(hidden)                     # logits (B, V)
```

## 학습 한 스텝 (수동 갱신 — 옵티마이저는 v0.1.2)

```python
logits = model(xs)                      # ① 순전파
loss   = F.cross_entropy(logits, ys)    # ② 손실 (softmax + NLL 을 한 번에)
for p in model.parameters(): p.grad = None   # ③ 이전 기울기 비우기 (backward 는 누적됨)
loss.backward()                         # ④ autograd 가 기울기 계산
with torch.no_grad():
    for p in model.parameters(): p -= LR * p.grad   # ⑤ 경사하강 1스텝
```

## 실행

> ⚠️ 이 버전부터 **PyTorch** 필요: `pip install torch`

```bash
python3 1.train/train.py     # 학습 → 0.model/model.pt + vocab.json + 1.train/loss.svg
python3 2.test/test.py       # 학습/검증 PPL + 이어쓰기 예시
```

- 학습 결과가 생기면 웹앱(`web_service`)에서 v0.1.0 을 골라 **이어쓰기**로 평가할 수 있어요.
- `0.model / 1.train / 2.test` 구조가 v0.0.x 와 같아 웹앱이 수정 없이 로드합니다.

## 성적 (산문 사전학습)

| | 값 |
|---|---|
| 학습 PPL | 6.07 |
| 검증 PPL | **6.47** |
| top-1 정확도 | 60.7% |
| 파라미터 | 232,456 |

- 카운트(2.65)에 **크게 뒤집니다** — 앞 1토큰만 보기 때문이에요.
- 문맥을 2토큰으로 맞추는 **v0.1.4** 에서 비로소 카운트를 넘어섭니다(2.62).
- 이 버전의 목적은 순위가 아니라 **'세기'에서 '학습'으로 원리를 바꾸는 것**입니다.

## 손실 곡선 — `1.train/loss.svg`

- 파란 실선 = **학습 손실**(왼쪽 축), 빨간 점선 = **검증 PPL**(오른쪽 축, 로그 눈금).
- 세로 점선 = 조기 종료가 **채택한 에폭**.
- 이 버전은 **full-batch 수동 SGD** 라 곡선이 완만하고, 상한(1,500 에폭)에서도 **아직 내려가는 중**이에요.
  → v0.1.1(미니배치)·v0.1.2(Adam)의 곡선과 비교하면 수렴 속도 차이가 바로 보입니다.
- matplotlib 없이 `save_loss_plot()` 이 SVG 를 직접 씁니다 (의존성은 torch 하나).
- 이후 모든 신경망 버전이 이 함수를 물려받아요.
- 학습 산출물이라 **커밋하지 않아요**(`.gitignore`) — `1.train/train.py` 를 돌리면 생깁니다.

## 조기 종료 (early stopping)

에폭 수를 손으로 맞추는 일을 없앱니다. 매 에폭 **학습 손실**과 **검증 PPL** 을 함께 재요.

- **멈출 때** — 둘 중 **하나라도** 최저를 갱신하면 계속, 둘 다 `PATIENCE` 에폭 정체하면 중단.
- **저장할 때** — **검증 PPL 이 가장 낮았던 에폭**의 가중치로 되돌려 저장 (= 일반화 최적점).
- 그래서 `EPOCHS` 는 맞춰야 하는 값이 아니라 넉넉한 **상한**이면 됩니다.

```python
m.EARLY_STOPPING = True    # False 면 EPOCHS 를 끝까지
m.PATIENCE = 20            # 몇 에폭까지 참을지 (전 버전 통일)
m.MIN_DELTA = 0.0          # 이만큼은 좋아져야 '개선'
```

> 💡 **왜 저장 기준이 검증인가.** 학습 손실은 계속 내려가지만 검증 PPL 은 어느 지점에서 바닥을 찍고
> 다시 올라가요 — 그 뒤는 과적합입니다. 마지막 에폭을 저장하면 그 과적합된 가중치를 배포하게 돼요.

> ⚠️ **PyTorch 본체에는 early stopping 이 없어요.** Lightning 의 콜백이나 Ignite 의 핸들러는 별도 패키지입니다.
> 의존성을 torch 하나로 유지하려고 `lm.py` 안에 15줄짜리 `EarlyStopping` 클래스로 직접 구현했어요
> (`ReduceLROnPlateau` 가 쓰는 patience 개념과 같은 방식).

## 계산 장치 (MPS / CPU)

```python
m.DEVICE = "auto"    # "auto" = MPS 있으면 사용, 없으면 CPU. "cpu"/"mps" 로 못박기 가능
```

## 다음 단계 — 신경망 시대 (v0.1.x, 6단계)

| 버전 | 한 걸음 |
|---|---|
| **v0.1.0 ← 현재** | 신경망 2층 MLP + autograd |
| v0.1.1 | 미니배치 학습 (`Dataset`/`DataLoader`) |
| v0.1.2 | 옵티마이저 (`torch.optim`) |
| v0.1.3 | 정규화·초기화 |
| v0.1.4 | 2토큰 문맥 (카운트와 같은 조건) |
| v0.1.5 | 기준선 대결 (캡스톤) |

> 이후 v0.2.x 에서 one-hot 을 **임베딩 벡터**로 바꿉니다 (Bengio 입력층).
