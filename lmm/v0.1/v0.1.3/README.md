# link_llm — 나만의 작은 언어 모델 (v0.1.3)

**정규화 · 초기화.** 모델·데이터·옵티마이저 사용법은 v0.1.2 그대로이고,
목표가 바뀌어요 — 손실을 더 낮추는 게 아니라 **일반화(검증 PPL) 개선**.

## v0.1.2 → v0.1.3, 무엇이 바뀌었나요?

> **과적합을 눌러 '처음 보는 것'에 강하게.** v0.0.9 의 학습 2.79 vs 검증 34.39(약 12배) 격차를 줄이는 게 목표예요.

| # | 더한 것 | 코드 | 왜 |
|---|---------|------|----|
| ① | **weight decay (L2)** | `Adam(..., weight_decay=1e-4)` | 큰 가중치에 벌점 → 확률 분포가 부드러워짐 → 검증 PPL↓ |
| ② | **초기화** | `nn.init.zeros_(model.linear.weight)` | 0에서 시작 = 처음엔 '균등 확률' |
| ③ | (선택) **label smoothing** | `F.cross_entropy(..., label_smoothing=0.1)` | 정답 과신↓ → 일반화 (기본 꺼둠) |

> 💡 이건 개수 세기 시대의 **v0.0.10 스무딩**과 같은 정신이에요 — "본 것에 과신하지 말자".

## 실행 방법

> ⚠️ **PyTorch** 필요.
> ```bash
> pip install torch
> ```

```bash
python3 3.train/train.py     # 정규화·초기화로 학습 → 2.models/model.json
python3 4.test/test.py       # 학습/검증 PPL 격차 + 대화 예시
```

`model.json` 이 생기면 웹앱(`web_service`)에서 v0.1.3 을 골라 **대화로 바로 평가**할 수 있어요.

## 완결성 — 웹앱에서 바로 평가

**학습 → 생성/대화 → PPL 측정**까지 한 버전에 완결돼요. 핵심 지표는
**학습 vs 검증 PPL 격차가 줄었나** — `WEIGHT_DECAY` 를 0 ↔ 1e-4 로 바꿔 비교해 보세요.

## 신경망 시대 진행 (v0.1.x, 5단계)

- v0.1.0 신경망 bigram + autograd (nn.Module)
- v0.1.1 Dataset · DataLoader 미니배치 학습
- v0.1.2 옵티마이저(`torch.optim`)
- **v0.1.3 정규화 · 초기화 ← 현재**
- v0.1.4 **기준선 대결(캡스톤)** — 웹앱에서 v0.0.9(카운트) vs 신경망 bigram PPL 비교
