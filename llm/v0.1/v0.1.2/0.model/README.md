# 0.model 폴더 (v0.1.2)

- `lm.py` : 이 버전 코드. v0.1.1 을 상속해 **파라미터 갱신을 `torch.optim` 으로** 바꿨어요.
- `model.pt` · `vocab.json` : 학습 결과 (`1.train/train.py` 실행 시 생성). **torch 환경에서 학습해야 생겨요.**

## 이 버전 모델의 특징

- **모델/데이터/확률 엔진:** v0.1.1 과 **동일** — `BigramModel`(nn.Linear 한 층) + `build_dataloader`(Dataset/DataLoader) 재사용.
- **바뀐 것 — '갱신'만 옵티마이저로:**
  - v0.1.1: `for p in model.parameters(): p -= LR * p.grad` (손으로)
  - v0.1.2: `optimizer.step()` (`torch.optim`)

## 표준 학습 루프 3줄

```python
optimizer = self.make_optimizer(model)     # 학습 전 한 번
...
for batch in loader:
    logits = model(batch["input"])
    loss = F.cross_entropy(logits, batch["target"])
    optimizer.zero_grad()                  # 이전 기울기 비우기 (p.grad = None 대신)
    loss.backward()                        # autograd
    optimizer.step()                       # 파라미터 갱신 (+ 모멘텀 / 적응 학습률)
```

## 옵티마이저 골라 쓰기 — `OPTIMIZER`

| 값 | 만드는 것 | 특징 |
|----|-----------|------|
| `"sgd"` | `torch.optim.SGD(params, lr)` | v0.1.1 과 같은 규칙(표준 API) |
| `"momentum"` | `SGD(params, lr, momentum=0.9)` | 관성 → 지그재그↓, 더 빨리 수렴 |
| `"adam"` | `torch.optim.Adam(params, lr)` | 파라미터마다 학습률 자동 (기본값) |

- **하이퍼파라미터:** `OPTIMIZER="adam"`, `LR=0.05`, `EPOCHS=100`, `BATCH_SIZE=64`.
  - ⚠️ **lr 은 옵티마이저마다 크게 달라요.** Adam 은 작게(0.01~0.1), SGD/momentum 은 크게(1~10).
  - **하이퍼파라미터는 `1.train/train.py` 에서 설정**해요. `OPTIMIZER` 를 바꿔가며 `2.test/test.py`(또는 학습 로그의 손실 곡선)로 수렴을 비교해 보세요.

## 저장 형식 — PyTorch 표준

| 파일 | 내용 |
|---|---|
| `model.pt` | 가중치 `state_dict` (`torch.save`). 항상 **CPU 텐서**로 저장 |
| `vocab.json` | `{ tokenizer, vocab }` — 어휘 목록 (위치 = 정수 인덱스) |

- 개수 세기(v0.0.x)의 `model.json`(개수 표) 과 다릅니다.
- 불러올 때 저장된 가중치 **모양에서 은닉 크기를 복원**해 신경망을 다시 만들고 `load_state_dict` 로 채워요.
