# link_lmm — 나만의 작은 언어 모델 (v0.2.0)

**임베딩 도입 (one-hot → `nn.Embedding`).** 신경망 시대 2막의 시작이에요.
토큰을 '독립된 칸(one-hot)'이 아니라 **작은 연속 벡터(임베딩)** 로 표현합니다.

## v0.1.5 → v0.2.0, 무엇이 바뀌었나요?

> **딱 '입력 표현' 하나.** 앞 2토큰 문맥·2층 MLP·정규화·저장방식은 v0.1.5 그대로,
> 입력만 one-hot(2V) → **임베딩 concat(2E)** 로 바꿔요.

| | v0.1.5 (one-hot) | v0.2.0 (임베딩) |
|---|---|---|
| 토큰 표현 | one-hot (V차원, 독립) | 임베딩 (E차원, 공유·학습) |
| 입력 크기 | 2V (예: 1586) | **2E (예: 64)** |
| 모델 | `Linear(2V,H)→tanh→Linear(H,V)` | `Embedding(V,E)` + `Linear(2E,H)→tanh→Linear(H,V)` |
| 새 표 | — | 임베딩 표 `C` (V×E, 함께 학습) |

## 핵심 통찰 — 임베딩 = one-hot 곱하기의 압축판

```python
nn.Embedding(V, E)(idx)  ==  F.one_hot(idx, V).float() @ C     # C: V×E
```

'원-핫 곱하기 가중치'가 곧 '표에서 한 줄 꺼내기(lookup)'예요. one-hot 의 첫 층을
임베딩 lookup 으로 바꾼 것 — 결과 형태는 같지만 **E ≪ V 로 압축·공유**된다는 게 핵심이에요.

## 왜 임베딩이 중요한가

- **one-hot 의 한계**: 토큰마다 독립된 칸 → "안녕 다음"에서 배운 걸 "반가워 다음"에 못 씀.
- **임베딩**: 비슷한 토큰이 비슷한 벡터를 갖도록 학습 → **문맥끼리 강점을 공유** →
  처음 보는 조합도 부드럽게 일반화 → 카운트의 백오프를 이길 진짜 무기.

## 구조 (Bengio 2003)

```
(앞앞, 앞) → 임베딩 lookup(각 E) → concat(2E) → fc1 → tanh → fc2 → logits(V)
```

## 실행 방법

> ⚠️ **PyTorch** 필요.
> ```bash
> pip install torch
> ```

```bash
python3 3.train/train.py     # 임베딩 MLP 학습 → 2.models/model.pt (+ vocab.json)
python3 4.test/test.py       # 학습/검증 PPL + 대화 예시
```

검증 PPL 이 **v0.1.5(one-hot 2토큰) 보다 내려가면** "임베딩의 공유 표현이 일반화에 도움",
더 내려가 **v0.0.9(카운트) 34.39** 아래면 드디어 신경망이 카운트를 넘어선 거예요.

## 저장 형식 (v0.1.x 와 동일 방식)

- `model.pt` : `torch.save(net.state_dict())` — 임베딩 표 `emb.weight`(V×E)도 함께 저장.
- `vocab.json` : `{ tokenizer, vocab }`.
- 불러올 때 `emb.weight` 모양에서 **임베딩 차원 E**, `fc1.weight` 에서 **은닉 H** 를 복원.

## 임베딩 시대 진행 (v0.2.x)

- **v0.2.0 임베딩 도입 ← 현재**
- v0.2.1 문맥 확장(`block_size`, 앞 N토큰)
- v0.2.2 초기화·정규화 · v0.2.3 일반화·튜닝
- v0.2.4 임베딩 시각화(PCA/t-SNE) · v0.2.5 **3자 비교(캡스톤)**

> 참고: 은닉층(2층 MLP)은 v0.1.x 에서 이미 도입했어요. v0.2.x 는 **표현(임베딩)과 문맥**에 집중.
