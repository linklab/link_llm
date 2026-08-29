# link_llm — 나만의 작은 언어 모델 (v0.1.2)

**`torch.optim` 옵티마이저로 학습.** 모델·데이터 파이프라인은 v0.1.1 그대로이고,
파라미터를 **손으로 갱신하던 것을 PyTorch 옵티마이저**로 바꿉니다.

## v0.1.1 → v0.1.2, 무엇이 바뀌었나요?

> **수동 갱신 → `optimizer.step()`.** `torch.optim` 이 파라미터 갱신을 대신 해줘요.
> 모멘텀·적응 학습률 같은 더 똑똑한 방법도 한 줄로 쓸 수 있어요.

| | v0.1.1 | v0.1.2 |
|---|---|---|
| 기울기 비우기 | `for p in ...: p.grad = None` | `optimizer.zero_grad()` |
| 갱신 | `with torch.no_grad(): p -= LR*p.grad` | `optimizer.step()` |
| 방법 | SGD(수동) | **SGD / momentum / Adam** 골라 쓰기 |

## 표준 학습 루프 (3줄)

```python
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # 학습 전 한 번
...
for batch in loader:
    logits = model(batch["input"])
    loss = F.cross_entropy(logits, batch["target"])
    optimizer.zero_grad()     # 이전 기울기 비우기
    loss.backward()           # autograd
    optimizer.step()          # 파라미터 갱신
```

## 옵티마이저 3종 비교

`1.train/train.py` 의 `OPTIMIZER` 를 바꿔 학습해 보세요:

- **`"sgd"`** — `p -= lr*grad`. v0.1.1 과 같은 규칙(이제 표준 API).
- **`"momentum"`** — 이전 방향을 관성처럼 누적 → 지그재그가 줄고 더 빨리 수렴.
- **`"adam"`** — 파라미터마다 학습률을 자동 조절 → lr 에 덜 민감, 보통 가장 빠름. **(기본값)**

> ⚠️ **lr 은 옵티마이저마다 크게 달라요.** Adam 은 작게(0.01~0.1), SGD/momentum 은 크게(1~10)로.

## 실행 방법

> ⚠️ **PyTorch** 필요.
> ```bash
> pip install torch
> ```

```bash
python3 1.train/train.py     # torch.optim 으로 학습 → 0.model/model.pt (+ vocab.json)
python3 2.test/test.py       # 학습/검증 PPL + 이어쓰기 예시
```

`model.pt` 가 생기면 웹앱(`web_service`)에서 v0.1.2 를 골라 **이어쓰기로 바로 평가**할 수 있어요.
(저장 형식이 이전과 같아 웹앱이 수정 없이 로드합니다.)

## 완결성 — 웹앱에서 바로 평가

**학습 → 생성/이어쓰기 → PPL 측정**까지 한 버전에 완결돼요. 산문 검증 PPL 이
v0.1.1(4.08)과 비슷한 **4.04** 로 나오면 "갱신 방법만 바꿔도 결과(모델)는 같다",
손실 곡선으로는 "옵티마이저가 수렴을 어떻게 바꾸나"를 봅니다.

## 신경망 시대 진행 (v0.1.x, 5단계)

- v0.1.0 신경망 bigram + autograd (nn.Module)
- v0.1.1 Dataset · DataLoader 미니배치 학습
- **v0.1.2 옵티마이저(`torch.optim`) ← 현재**
- v0.1.3 정규화·초기화 → v0.1.4 **기준선 대결(캡스톤)**: 웹앱에서 v0.0.9(카운트) vs 신경망 bigram PPL 비교
