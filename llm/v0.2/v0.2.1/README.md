# link_llm — 나만의 작은 언어 모델 (v0.2.1)

**문맥 확장 (`block_size`).** 2토큰 고정을 **앞 N토큰**으로 열어요.
각 토큰 임베딩을 이어붙여 MLP 에 넣는 = 진짜 **Bengio(2003) 입력층**.

## 한 걸음 — 문맥 길이를 설정값으로

| | v0.2.0 | v0.2.1 |
|---|---|---|
| 문맥 | 앞 2토큰 (고정) | **앞 N토큰** (`BLOCK_SIZE`) |
| 입력 | (B, 2) | (B, N) |
| 모델 | `Linear(2E, H)` | `Linear(N·E, H)` |
| 임베딩 concat | 2개 | **N개** (`emb(x).flatten(1)`) |

`BLOCK_SIZE=2` 로 두면 v0.2.0 과 **완전히 동일**합니다 (실측 검증 PPL 2.9446 일치).

## 구조

```
앞 N토큰 → 각 임베딩(E) → 이어붙임(N·E) → fc1 → tanh → fc2 → logits(V)
```

```python
class BlockModel(nn.Module):
    def __init__(self, V, H, E, block_size):
        self.emb = nn.Embedding(V, E)
        self.fc1 = nn.Linear(block_size * E, H)
        self.fc2 = nn.Linear(H, V)
    def forward(self, x):          # x: (B, N) 인덱스
        e = self.emb(x).flatten(1) # (B, N, E) → (B, N·E)
        return self.fc2(torch.tanh(self.fc1(e)))
```

## 문장 맨 앞 — `<PAD>`

- N토큰이 다 안 차는 문장 앞부분은 `<PAD>`(어휘 0번)로 채워요.
- 예) `ids=[a,b,c]`, N=3 → `(p,p,a)→b`, `(p,a,b)→c`, `(a,b,c)→<END>`

## 구현 — 최소 override

- 새로 정의: `build_net`(BlockModel) · `make_pairs`(N토큰) · `_context_ids`(앞 N토큰 인덱스 목록) · `load`(모양에서 N·E·H 복원)
- 그대로 상속: 임베딩 · 정규화 · 저장 · 대화 · PPL
- 새 하이퍼파라미터: **`BLOCK_SIZE`**(기본 3)

## 성적 (산문 사전학습) — 숫자를 조심해서 읽으세요

기본 설정(N=3)의 검증 PPL 은 **3.37** 입니다.

> ⚠️ **이 숫자는 '문맥을 넓힌 결과'가 아니에요.**
> 이 버전은 `BLOCK_SIZE` 말고도 `EMBED` 256→32 · `HIDDEN` 256→128 · `LR` · `INIT` ·
> `LABEL_SMOOTHING` 까지 **5가지를 함께** 바꿨습니다.

**문맥만 떼어 재보면** (루트 `ablation_block_size.py`):

| 설정 묶음 | N=2 | N=3 | N 효과 |
|---|---|---|---|
| E=256, H=256 | 2.9446 | **2.9182** | **−0.026** (좋아짐) |
| E=32, H=128 | 3.3651 | 3.3835 | +0.018 (나빠짐) |

- **문맥 효과 ±0.02** — 부호도 설정에 따라 뒤집히는 **잡음** 수준.
- **용량 효과 0.43** — `EMBED`·`HIDDEN` 을 줄인 것. 약 **20배**.
- ⇒ v0.2.1 이 뒤진 원인은 **문맥이 아니라 용량**입니다.

> 💡 **한 번에 하나만 바꿔야 원인을 말할 수 있어요** — '버전 = 개념 한 걸음' 원칙이 바로 그 이야기입니다.

## 실행

> ⚠️ **PyTorch** 필요: `pip install torch`

```bash
python3 1.train/train.py     # N토큰 문맥 학습 → 0.model/model.pt (+ vocab.json)
python3 2.test/test.py       # 학습/검증 PPL + 이어쓰기 예시
```

- `BLOCK_SIZE` 를 2·3·4 로 바꿔 비교해 보세요 — 단, **`EMBED`·`HIDDEN` 은 고정한 채로**.

## 임베딩 시대 진행 (v0.2.x)

| 버전 | 한 걸음 |
|---|---|
| v0.2.0 | 임베딩 도입 |
| **v0.2.1 ← 현재** | 문맥 확장 (`block_size`) |
| v0.2.2 | 초기화·정규화 |
| v0.2.3 | 일반화·튜닝 |
| v0.2.4 | 임베딩 시각화 |
| v0.2.5 | 3자 비교 (캡스톤) |
