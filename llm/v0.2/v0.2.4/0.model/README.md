# 0.model 폴더 (v0.2.4)

- `lm.py` : 이 버전 코드. v0.2.3 을 상속하고 **모델은 하나도 바꾸지 않아요** — 분석 함수만 추가.
- `model.pt` · `vocab.json` : 학습 결과 (`1.train/train.py` 실행 시 생성). **torch 환경에서 학습해야 생겨요.**

## 이 버전 모델의 특징

**목표:** 학습된 **임베딩 표를 들여다보는 것**. 모델·학습·저장은 v0.2.3 그대로입니다.

| 추가된 함수 | 하는 일 |
|---|---|
| `embedding_matrix()` | `emb.weight` (V×E) 를 CPU 텐서로 |
| `token_frequency(sentences)` | 토큰별 등장 횟수 (자주 쓰는 토큰만 찍으려고) |
| `plot_tokens(sentences)` | 산점도에 찍을 토큰 인덱스 (빈도순) |
| `nearest(token, k)` | 코사인 유사도로 가장 가까운 k개 |
| `pca_2d(idx)` | 2차원 좌표 + **설명된 분산 비율** |
| `tsne_2d(idx)` | 2차원 좌표 + 최종 KL |
| `save_embedding_plot(...)` | 산점도를 **SVG 로 직접** 저장 |

## 왜 CPU 에서 분석하나

`embedding_matrix()` 가 텐서를 CPU 로 옮깁니다.

- 한 번만 하는 계산이라 GPU 이득이 없어요.
- MPS 는 `linalg.svd` 를 **CPU 로 되돌리면서 경고**를 냅니다 — 처음부터 CPU 로 두는 게 깔끔해요.

## PCA — SVD 한 번

```python
Ec = E - E.mean(dim=0, keepdim=True)          # 중심화
_, S, Vh = torch.linalg.svd(Ec, full_matrices=False)
coords = Ec @ Vh[:2].T                         # 상위 2개 주성분에 투영
ratio  = (S[:2]**2).sum() / (S**2).sum()       # 설명된 분산
```

중심화 후 SVD 를 하면 **특이벡터가 곧 주성분**이에요. 상위 2개에 투영하면
"분산을 가장 많이 남기는" 2차원 그림이 됩니다.

## t-SNE — 두 단계

**① 고차원 유사도 P** — 각 점마다 가우시안 폭을 **이분 탐색**으로 정해요.
이웃 수가 대략 `perplexity` 가 되도록 맞추는 것 (조밀한 곳은 좁게, 성긴 곳은 넓게).

**② 저차원 좌표 Q 최적화** — t-분포 커널을 쓰고 KL(P‖Q) 를 **autograd 로** 줄여요.

- t-분포는 꼬리가 두꺼워서 멀리 있는 점들이 서로 밀어낼 여지를 줍니다 — 이게 t-SNE 의 't'.
- 초반 250스텝은 **early exaggeration**(P 를 4배)로 덩어리를 먼저 뭉치게 해요.
  그 구간의 KL 값은 스케일이 달라 **올라가는 것처럼 보이는 게 정상**입니다.

## 산점도 SVG

`save_embedding_plot()` 이 좌표를 받아 SVG 를 직접 씁니다 (matplotlib 없음).

- 점마다 **토큰 이름을 옆에** 써서 "무엇이 무엇 옆에 있나" 를 바로 읽을 수 있어요.
- SVG 는 벡터라 확대해도 글자가 안 깨집니다.
- 출력은 `2.test/embedding_pca.svg` · `2.test/embedding_tsne.svg` (커밋하지 않는 산출물).

## 시각화 설정

| 이름 | 기본값 | 뜻 |
|---|---|---|
| `PLOT_TOKENS` | 200 | 산점도에 찍을 토큰 수 (빈도순) |
| `TSNE_PERPLEXITY` | 20.0 | 이웃 폭 — 낮으면 지역, 높으면 전역 구조 |
| `TSNE_STEPS` | 500 | 경사하강 반복 수 |
| `TSNE_LR` | 100.0 | t-SNE 학습률 |
| `SEED` | 1234 | 시각화 재현용 (학습 때는 `train.py` 값이 덮어씀) |

## 저장 형식 — v0.2.3 과 동일

| 파일 | 내용 |
|---|---|
| `model.pt` | 가중치 `state_dict` (weight tying 이면 `fc2.weight` 는 빼고 저장) |
| `vocab.json` | `{ tokenizer, vocab }` — 어휘 목록 (0번 = `<PAD>`) |
