# link_lmm — 나만의 작은 언어 모델 (v0.1.0)

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

### 한 스텝의 학습 (핵심 5줄)

```python
logits = W[xs]                     # 순전파: one-hot(x) @ W == W[x]
loss   = F.cross_entropy(logits, ys)   # softmax + NLL 을 한 번에
W.grad = None
loss.backward()                    # autograd 가 기울기 계산
with torch.no_grad():
    W -= LR * W.grad               # 경사하강 1스텝 (수동 갱신)
```

> **핵심 통찰:** 학습이 끝난 신경망 bigram 은 **개수 bigram 과 사실상 같아져요.**
> 경사하강이 결국 "그 문맥에서의 다음 토큰 등장 비율"을 재현하거든요.

## 실행 방법

> ⚠️ 이 버전부터 **PyTorch** 가 필요해요. torch 를 설치할 수 있는 파이썬 환경에서 실행하세요.
> ```bash
> pip install torch
> ```

```bash
python3 3.train/train.py     # 학습 → 2.models/model.json 생성
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
