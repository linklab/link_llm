# link_llm — 나만의 작은 언어 모델 (v0.1.2)

**옵티마이저 도입.** 모델·데이터 파이프라인은 v0.1.1 그대로이고,
파라미터 갱신을 **손으로 하던 것을 `torch.optim`** 에 맡깁니다.

## 한 걸음 — 수동 갱신 → `optimizer.step()`

| | v0.1.1 | v0.1.2 |
|---|---|---|
| 기울기 비우기 | `for p in ...: p.grad = None` | `optimizer.zero_grad()` |
| 갱신 | `with torch.no_grad(): p -= LR*p.grad` | `optimizer.step()` |
| 방법 | SGD (수동) | **SGD / momentum / Adam** 골라 쓰기 |

## 표준 학습 루프 (3줄)

```python
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # 학습 전 한 번
...
for batch in loader:
    logits = model(batch["input"])
    loss = F.cross_entropy(logits, batch["target"])
    optimizer.zero_grad()     # ① 이전 기울기 비우기
    loss.backward()           # ② autograd
    optimizer.step()          # ③ 파라미터 갱신
```

## 옵티마이저 3종

`1.train/train.py` 의 `OPTIMIZER` 를 바꿔 학습해 보세요.

| 값 | 규칙 | 특징 |
|---|---|---|
| `"sgd"` | `p -= lr*grad` | v0.1.1 과 같은 규칙 (이제 표준 API) |
| `"momentum"` | 이전 방향을 관성처럼 누적 | 지그재그가 줄고 더 빨리 수렴 |
| `"adam"` | 파라미터마다 학습률 자동 조절 | lr 에 덜 민감, 보통 가장 빠름 **(기본값)** |

> ⚠️ **lr 은 옵티마이저마다 크게 달라요.** Adam 은 작게(0.01~0.1), SGD/momentum 은 크게(1~10).

## 실행

> ⚠️ **PyTorch** 필요: `pip install torch`

```bash
python3 1.train/train.py     # torch.optim 으로 학습 → 0.model/model.pt (+ vocab.json)
python3 2.test/test.py       # 학습/검증 PPL + 이어쓰기 예시
```

저장 형식이 이전과 같아 웹앱(`web_service`)이 수정 없이 로드합니다.

## 성적 (산문 사전학습)

| 버전 | 검증 PPL | 채택 에폭 |
|---|---|---|
| v0.1.1 (수동 SGD) | 4.08 | 185 |
| **v0.1.2 (Adam)** | **4.04** | **18** |

- PPL 은 거의 같아요 — **갱신 방법을 바꿔도 도달하는 모델은 비슷**합니다.
- 크게 다른 건 **속도**: Adam 은 18에폭 만에 검증 최적점에 도달해요(v0.1.1 은 185에폭).
- 옵티마이저의 값어치는 최종 성능보다 **수렴 속도**에서 드러납니다 — `loss.svg` 곡선을 비교해 보세요.

## 신경망 시대 진행 (v0.1.x, 6단계)

| 버전 | 한 걸음 |
|---|---|
| v0.1.0 | 신경망 2층 MLP + autograd |
| v0.1.1 | 미니배치 학습 |
| **v0.1.2 ← 현재** | 옵티마이저 (`torch.optim`) |
| v0.1.3 | 정규화·초기화 |
| v0.1.4 | 2토큰 문맥 |
| v0.1.5 | 기준선 대결 (캡스톤) |
