# 0.model 폴더 (v0.2.2)

- `lm.py` : 이 버전 코드. v0.2.1 을 상속해 **초기화·정규화 도구**를 더했어요.
- `model.pt` / `vocab.json` : 학습 결과 (`1.train/train.py` 실행 시 생성). **torch 필요.**

## 이 버전 모델의 특징

**목표:** 깊은 모델을 안정적으로 학습하는 **초기화·정규화 도구**를 갖추기 (용량·문맥은 v0.2.1 그대로).

| # | 항목 | v0.2.1 | v0.2.2 |
|---|------|--------|--------|
| ① | 모델 | `BlockModel` | `NormModel` (fc1 뒤 BatchNorm **토글**) |
| ② | 초기화 | 기본 | `init_model`: **Kaiming/Xavier**(tanh 이득) |
| ③ | 정규화 | wd 0 | weight_decay 1e-4 |
| ④ | 하이퍼파라미터 | — | **`USE_BN`** 추가 (기본 False) |

## 코드 요점

```python
class NormModel(nn.Module):
    def __init__(self, V, hidden, embed, block_size, use_bn=False):
        self.emb = nn.Embedding(V, embed)
        self.fc1 = nn.Linear(block_size*embed, hidden, bias=not use_bn)  # BN 있으면 bias 불필요
        self.bn  = nn.BatchNorm1d(hidden) if use_bn else None
        self.fc2 = nn.Linear(hidden, V)
    def forward(self, x):
        z = self.fc1(self.emb(x).flatten(1))
        if self.bn is not None: z = self.bn(z)
        return self.fc2(torch.tanh(z))
```

- **초기화**: `nn.init.xavier_normal_(fc1.weight, gain=calculate_gain('tanh'))` — tanh 앞 층에 맞는 크기.
- **불러오기**: 저장된 state_dict 에서 **E·N·H + BatchNorm 유무(`bn.weight` 존재)** 까지 되살려 정확히 복원.

## 상속(그대로인 것)
- 임베딩·문맥(block_size)·`make_pairs`·`_context_ids`·`<PAD>` — v0.2.1/v0.1.4
- 옵티마이저·label smoothing — v0.1.2/1.3 · DataLoader — v0.1.1
- 학습 루프·**조기 종료**·저장·대화·PPL — v0.1.0

## 평가 포인트
`2.test/test.py` 검증 PPL 이 **~2.65** 로 나오면 정상이에요 (v0.2.1 2.37 보다 높아요) —
쉬운 산문에선 이 도구들의 이득이 작거든요(진가는 더 깊고 큰 모델에서).
`USE_BN=True/False`, `INIT=kaiming/default/zeros` 를 바꿔가며 각 도구의 효과를 관찰해 보세요.
