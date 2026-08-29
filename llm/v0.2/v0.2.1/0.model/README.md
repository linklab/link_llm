# 0.model 폴더 (v0.2.1)

- `lm.py` : 이 버전 코드. v0.2.0 을 상속해 **문맥 2토큰 → 앞 N토큰(`BLOCK_SIZE`)** 으로 넓혔어요.
- `model.pt` / `vocab.json` : 학습 결과 (`1.train/train.py` 실행 시 생성). **torch 필요.**

## 이 버전 모델의 특징

**목표:** 2토큰 고정을 **앞 N토큰(`BLOCK_SIZE`)** 으로 열기.

v0.2.0 은 앞 2토큰 고정. v0.2.1 은 `BLOCK_SIZE`(=N)로 **앞 N토큰**을 보게 해요.
각 토큰 임베딩을 이어붙여 MLP 입력으로 = Bengio(2003) 입력층. 문맥이 길수록 더 먼
과거를 참고하지만, 입력·파라미터가 커져 과적합 위험도 함께 커져요(그래서 N 을 튜닝).

## 무엇을 바꿨나 (v0.2.0 → v0.2.1)

| # | 항목 | v0.2.0 | v0.2.1 |
|---|------|--------|--------|
| ① | 모델 | `EmbeddingModel`(2토큰) | `BlockModel`(N토큰) |
| ② | forward | `emb(prev2),emb(prev1)` concat | `emb(x).flatten(1)` (N개 concat) |
| ③ | 입력 크기 | 2·E | **N·E** |
| ④ | 데이터 | `(prev2,prev1)→next` | `앞 N토큰 → next` (`make_pairs` 재정의) |
| ⑤ | 하이퍼파라미터 | `EMBED` | + **`BLOCK_SIZE`** |

## 불러오기 — 저장된 모양에서 N·E·H 복원

```python
E = emb.weight.shape[1]              # 임베딩 차원
N = fc1.weight.shape[1] // E         # 문맥 길이 (N·E ÷ E)
H = fc1.weight.shape[0]              # 은닉
```
그래서 `vocab.json` 엔 어휘만 있어도 신경망을 정확히 복원해요.

## 그대로인 것 (상속)

- 임베딩(`nn.Embedding`) 개념 — v0.2.0
- 정규화·초기화·label smoothing — v0.1.3 · DataLoader — v0.1.1 · `<PAD>` — v0.1.4
- 학습 루프·저장(`model.pt`)·대화·퍼플렉서티 — v0.1.0

## 평가 포인트

`2.test/test.py` 의 검증 PPL 이 v0.2.0(2토큰) 보다 낮아지는지 확인. `BLOCK_SIZE` 를
2·3·4 로 바꿔 문맥 길이와 일반화의 관계를 관찰해 보세요.
