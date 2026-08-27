# link_llm — 나만의 작은 언어 모델 (v0.1.1)

**PyTorch `Dataset` + `DataLoader` 로 학습.** 모델은 v0.1.0(신경망 bigram) 그대로이고,
학습 데이터를 공급하는 방식을 강의 자료(`03.real_world_data_to_tensors`)의 표준 방식으로 바꿨어요.

## v0.1.0 → v0.1.1, 무엇이 바뀌었나요?

> **수동 미니배치 → `Dataset` + `DataLoader`.** 데이터를 `Dataset` 으로 감싸면,
> `DataLoader` 가 매 에폭 **셔플**하고 **미니배치**로 묶어 줍니다. 모델·확률 엔진·저장 형식은 v0.1.0과 **동일**.

| | v0.1.0 | v0.1.1 |
|---|---|---|
| 한 스텝에 쓰는 데이터 | 전체(full-batch) | 배치(64) |
| 배치·셔플 | 직접 (`randperm`) | **DataLoader 가 대신** |
| 데이터 표현 | 텐서 두 개 | **`Dataset` (`{'input','target'}`)** |

## 강의 자료 관례 그대로

```python
# 2.models/dataloader.py — 데이터셋 + 로더 생성을 여기에 모아요
from torch.utils.data import Dataset, DataLoader

class BigramDataset(Dataset):
    def __init__(self, xs, ys):
        self.data, self.target = xs, ys
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {"input": self.data[idx], "target": self.target[idx]}

def build_dataloader(xs, ys, batch_size, shuffle=True):
    dataset = BigramDataset(xs, ys)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

# 2.models/lm.py — 학습에서는 한 줄로 데이터로더를 만들어요
torch.manual_seed(SEED)
model  = BigramModel(V)                            # v0.1.0 의 nn.Module 모델 재사용
loader = build_dataloader(xs, ys, batch_size=64, shuffle=True)
for batch in loader:
    logits = model(batch["input"])                 # 순전파 (forward)
    loss = F.cross_entropy(logits, batch["target"])
    for p in model.parameters(): p.grad = None
    loss.backward()                                # autograd
    with torch.no_grad():
        for p in model.parameters(): p -= LR * p.grad   # 배치마다 갱신
```

- **`BigramModel`** — v0.1.0 의 `nn.Module` 모델(`nn.Linear` 한 층)을 그대로 재사용. `model(x)` 로 순전파.
- **`BigramDataset` + `build_dataloader`** — 데이터셋 클래스와 로더 생성 헬퍼를 **`2.models/dataloader.py`** 에 분리. `Dataset` 은 `__getitem__` 이 **`{'input','target'}` 딕셔너리**를 반환.
- **`DataLoader`** — 셔플 + 배치를 자동으로. lm.py 는 `loader = build_dataloader(xs, ys, batch_size)` 한 줄만 부릅니다. (`torch.manual_seed(SEED)` 로 재현 가능)
- 앞으로(v0.1.x~) 이 데이터 파이프라인을 계속 씁니다.

## 실행 방법

> ⚠️ **PyTorch** 필요. torch 되는 파이썬 환경에서 실행하세요.
> ```bash
> pip install torch
> ```

```bash
python3 3.train/train.py     # Dataset/DataLoader 로 학습 → 2.models/model.json
python3 4.test/test.py       # 학습/검증 PPL + 대화 예시
```

`model.json` 이 생기면 웹앱(`web_service`)에서 v0.1.1 을 골라 **대화로 바로 평가**할 수 있어요.
(저장 형식이 v0.1.0 과 같아 웹앱이 수정 없이 로드합니다.)

## 완결성 — 웹앱에서 바로 평가

**학습 → 생성/대화 → PPL 측정**까지 한 버전에 완결돼요. PPL 이 v0.1.0 과 비슷하게 나오면
"데이터 공급 방식만 바꿔도 결과(모델)는 같다"가 확인됩니다.

## 신경망 시대 진행 (v0.1.x, 5단계)

- v0.1.0 신경망 bigram + autograd
- **v0.1.1 Dataset · DataLoader 미니배치 학습 ← 현재**
- v0.1.2 옵티마이저(`torch.optim`) → v0.1.3 정규화·초기화
- v0.1.4 **기준선 대결(캡스톤)** — 웹앱에서 v0.0.9(카운트) vs 신경망 bigram PPL 비교
