# 0.model 폴더 (v0.1.1)

- `lm.py` : 이 버전 코드. v0.1.0(신경망 bigram)을 상속해 **학습 파이프라인을 PyTorch `Dataset` + `DataLoader`** 로 바꿨어요.
- `model.json` : 학습 결과 (`1.train/train.py` 실행 시 생성). **torch 환경에서 학습해야 생겨요.**

## 이 버전 모델의 특징

- **모델/확률 엔진:** v0.1.0 과 **동일** — `nn.Module` 모델 `BigramModel`(`nn.Linear` 한 층)을 그대로 재사용해 `model(x)` 로 순전파.
- **바뀐 것 — 데이터 공급 방식:**
  - v0.1.0: 전체 데이터를 한 번에(full-batch).
  - v0.1.1: **`Dataset` + `DataLoader`** 로 감싸 DataLoader 가 **셔플 + 미니배치**를 대신.
- **하이퍼파라미터:** `LR=10.0`, `EPOCHS=25`, `BATCH_SIZE=64`.
- **손실 곡선:** 에폭별 평균 손실을 `self.losses` 에 기록.

## 강의 자료(03.real_world_data_to_tensors) 관례를 그대로

```python
# 0.model/dataloader.py — 데이터셋 + 로더 생성을 모아둠
from torch.utils.data import Dataset, DataLoader

class BigramDataset(Dataset):
    def __init__(self, xs, ys):
        self.data = xs        # 앞 토큰 (입력)
        self.target = ys      # 다음 토큰 (정답)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {"input": self.data[idx], "target": self.target[idx]}   # ← 딕셔너리 반환

def build_dataloader(xs, ys, batch_size, shuffle=True):
    dataset = BigramDataset(xs, ys)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

# 0.model/lm.py 에서는 한 줄로:
loader = build_dataloader(xs, ys, batch_size=64, shuffle=True)
for batch in loader:
    bx, by = batch["input"], batch["target"]   # 배치로 나옴
    ...
```

- **`0.model/dataloader.py`** 에 `BigramDataset` + `build_dataloader` 를 모아 뒀어요.
- `Dataset` 세 메서드: `__init__`(data·target 준비) / `__len__`(개수) / `__getitem__`(**한 샘플을 `{'input':..,'target':..}` 로**).
- `DataLoader` 가 매 에폭 **셔플**하고 **배치 크기**로 묶어 줘요 (v0.1.0 의 수동 `randperm`/슬라이싱을 대체).
- 학습 시작에 `torch.manual_seed(SEED)` 를 한 번 불러 초기화·셔플을 **재현 가능**하게.

## model.json 형식

v0.1.0 과 **똑같아요** — `{ type: "neural_bigram", tokenizer, vocab, W }`.
데이터 공급 방식이 달라도 저장되는 건 학습된 `W` + 어휘라, 웹앱/평가가 그대로 로드합니다.
`2.test/test.py` 의 PPL 이 v0.1.0 과 비슷하게 나오면 "파이프라인만 바꿔도 결과는 같다"가 확인돼요.
