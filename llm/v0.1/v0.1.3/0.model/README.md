# 0.model 폴더 (v0.1.3)

- `lm.py` : 이 버전 코드. v0.1.2 를 상속해 **정규화 + 초기화**를 더했어요.
- `model.json` : 학습 결과 (`1.train/train.py` 실행 시 생성). **torch 환경에서 학습해야 생겨요.**

## 이 버전의 목표 — 손실↓ 이 아니라 일반화↑

지금 프로즈 데이터는 쉬워서 격차가 작아요 — v0.0.9 에서 **학습 PPL 2.49 vs 검증 PPL 2.96**.
그래도 신경망을 그냥 두면 "본 것만 잘 맞히는" 과적합이 생기고, **진짜 격차는 v0.5 SFT** 에서 벌어져요.
v0.1.3 은 **과신을 눌러** 처음 보는 조합에도 덜 틀리게 만드는 도구를 미리 익힙니다.
(개수 세기 시대의 **v0.0.10 스무딩**과 같은 정신.)

## 무엇을 더했나 (v0.1.2 → v0.1.3)

| # | 항목 | 코드 | 효과 |
|---|------|------|------|
| ① | **weight decay (L2)** | `Adam(params, lr, weight_decay=1e-4)` | 큰 가중치에 벌점 → 분포 부드럽게 → 검증 PPL↓ |
| ② | **초기화** | `nn.init.zeros_(model.linear.weight)` | 0에서 시작 = 처음엔 균등 확률(라플라스 스무딩의 출발점) |
| ③ | (선택) **label smoothing** | `F.cross_entropy(..., label_smoothing=0.1)` | 정답 과신을 눌러 일반화 (기본 0.0=끔) |

- **하이퍼파라미터** (`1.train/train.py` 에서 설정): `WEIGHT_DECAY=1e-4`, `INIT="zeros"`, `LABEL_SMOOTHING=0.0` (+ `OPTIMIZER/LR/EPOCHS/BATCH_SIZE`).
- `WEIGHT_DECAY=0` 이면 v0.1.2 와 동일. 값을 바꿔가며 **학습 vs 검증 PPL 격차**가 어떻게 변하는지 보세요.
- 수치 안정성은 이미 `F.cross_entropy`(내부 log-sum-exp)가 담당.

## 그대로인 것 / 저장 형식

- `BigramModel`(nn.Module) · `build_dataloader`(Dataset/DataLoader) · `torch.optim` · 대화 · PPL — 전부 상속.
- `model.json` 은 v0.1.0~ 과 **동일** (`{ type, tokenizer, vocab, W }`). 웹앱/평가가 그대로 로드.
- 평가 핵심: `2.test/test.py` 의 **학습/검증 PPL 격차**가 줄었나 (그리고 v0.1.4 에서 v0.0.9 와 비교).
