# link_llm — 나만의 작은 언어 모델 (v0.1.1)

**미니배치 학습.** 모델은 v0.1.0(2층 MLP) 그대로이고, **데이터를 공급하는 방식**만 바꿔요.

## 한 걸음 — full-batch → `Dataset` + `DataLoader`

| | v0.1.0 | v0.1.1 |
|---|---|---|
| 한 스텝에 쓰는 데이터 | 전체(full-batch) | **배치(64)** |
| 에폭당 갱신 횟수 | 1번 | 배치 수만큼 (약 1,137번) |
| 데이터 순서 | 고정 | **매 에폭 셔플** |
| 배치·셔플 담당 | 직접 | **`DataLoader`** |

모델·확률 엔진·저장 형식은 v0.1.0 과 **동일**합니다.

## 코드

```python
# data/dataloader.py — 데이터셋 + 로더 생성을 한곳에
class BigramDataset(Dataset):
    def __init__(self, xs, ys):
        self.data, self.target = xs, ys
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {"input": self.data[idx], "target": self.target[idx]}   # 딕셔너리로 반환

def build_dataloader(xs, ys, batch_size, shuffle=True):
    return DataLoader(dataset=BigramDataset(xs, ys), batch_size=batch_size, shuffle=shuffle)

# 0.model/lm.py — 학습 루프
torch.manual_seed(SEED)
model  = self.make_net(V, HIDDEN)                   # v0.1.0 의 2층 MLP 재사용
loader = build_dataloader(xs, ys, batch_size=64, shuffle=True)
for batch in loader:
    logits = model(batch["input"])
    loss = F.cross_entropy(logits, batch["target"])
    for p in model.parameters(): p.grad = None
    loss.backward()
    with torch.no_grad():
        for p in model.parameters(): p -= LR * p.grad   # 배치마다 갱신
```

## 역할 분담

- **`Dataset`** — "인덱스 하나 → 샘플 하나". `__getitem__` 이 `{'input','target'}` 딕셔너리를 반환.
- **`DataLoader`** — "섞고 묶어서 배치로". 같은 키끼리 쌓아(collate) 배치도 같은 키를 가져요.
- **`build_dataloader`** — 위 둘을 한 줄로 조립.
- 파일은 `data/dataloader.py` 에 분리 — v0.1.x 이후 모든 버전이 공용합니다.
- `torch.manual_seed(SEED)` 로 셔플까지 재현 가능.

## 실행

> ⚠️ **PyTorch** 필요: `pip install torch`

```bash
python3 1.train/train.py     # Dataset/DataLoader 로 학습 → 0.model/model.pt (+ vocab.json)
python3 2.test/test.py       # 학습/검증 PPL + 이어쓰기 예시
```

저장 형식이 v0.1.0 과 같아 웹앱(`web_service`)이 수정 없이 로드합니다.

## 성적 (산문 사전학습)

| 버전 | 검증 PPL | top-1 |
|---|---|---|
| v0.1.0 (full-batch) | 6.47 | 56.0% |
| **v0.1.1 (미니배치)** | **4.78** | 57.9% |

- **모델이 같은데 PPL 이 절반 아래로** 내려갔어요.
- 원인은 구조가 아니라 **공급 방식** — 에폭당 갱신이 1번에서 1,137번으로 늘고, 매 에폭 순서가 섞입니다.
- 교훈: **같은 모델도 어떻게 학습시키느냐로 성능이 크게 달라진다.**
- 아직 카운트(2.65)에는 크게 뒤집니다 — 앞 1토큰만 보기 때문(문맥 확장은 v0.1.4).

## 신경망 시대 진행 (v0.1.x, 6단계)

| 버전 | 한 걸음 |
|---|---|
| v0.1.0 | 신경망 2층 MLP + autograd |
| **v0.1.1 ← 현재** | 미니배치 학습 (`Dataset`/`DataLoader`) |
| v0.1.2 | 옵티마이저 (`torch.optim`) |
| v0.1.3 | 정규화·초기화 |
| v0.1.4 | 2토큰 문맥 |
| v0.1.5 | 기준선 대결 (캡스톤) |
