# 0.model 폴더 (v0.2.0)

- `lm.py` : 이 버전 코드. v0.1.5 를 상속해 **입력을 one-hot → 임베딩**으로 바꿨어요.
- `model.pt` / `vocab.json` : 학습 결과 (`1.train/train.py` 실행 시 생성). **torch 필요.**

## 이 버전 모델의 특징

**목표:** 토큰 표현을 one-hot → **임베딩**으로 바꿔 비슷한 토큰끼리 강점을 공유하기.

one-hot 은 토큰마다 독립된 칸이라 문맥끼리 정보를 못 나눠요. 임베딩은 토큰을 작은
연속 벡터로 학습해, 비슷한 토큰이 비슷한 벡터를 갖게 됩니다 → 일반화의 열쇠.

## 무엇을 바꿨나 (v0.1.5 → v0.2.0)

| # | 항목 | v0.1.5 | v0.2.0 |
|---|------|--------|--------|
| ① | 입력 표현 | `F.one_hot`(2V) | `nn.Embedding(V, E)` lookup → concat(2E) |
| ② | 모델 | `ContextModel` | `EmbeddingModel` (emb + fc1 + fc2) |
| ③ | 새 파라미터 | — | 임베딩 표 `emb.weight` (V×E) |
| ④ | 하이퍼파라미터 | — | `EMBED`(E) 추가 |

`build_net` 과 `load` 만 override 하고, **나머지는 전부 상속**해요:

- 2토큰 문맥(`make_pairs`·`_context_ids`·`<PAD>`) — v0.1.4
- 정규화·초기화(`make_optimizer`·`init_model`·label smoothing) — v0.1.3
- 옵티마이저 — v0.1.2 · DataLoader — v0.1.1
- 학습 루프·저장/로드·대화·퍼플렉서티 — v0.1.0

## 핵심 등식

```python
nn.Embedding(V, E)(idx)  ==  F.one_hot(idx, V).float() @ C     # C = emb.weight (V×E)
```

## 불러오기 — 저장된 모양에서 차원 복원

`load` 는 `emb.weight` 에서 **E**, `fc1.weight` 에서 **H** 를 읽어 신경망을 다시 만든 뒤
`load_state_dict` 로 채워요. 그래서 vocab.json 엔 어휘만 있어도 됩니다.

## 평가 포인트

`2.test/test.py` 의 검증 PPL 이 v0.1.5(one-hot) 보다 낮아지는지, 나아가 v0.0.9(카운트,
2.65)를 넘는지 보세요. 넘으면 "임베딩으로 신경망이 카운트를 이겼다".
