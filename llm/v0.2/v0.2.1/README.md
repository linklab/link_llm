# link_llm — 나만의 작은 언어 모델 (v0.2.1)

**문맥 확장 (`block_size`).** v0.2.0 은 앞 **2토큰** 고정이었는데, v0.2.1 은 문맥 길이를
**`BLOCK_SIZE`(=N)** 로 열어 **앞 N토큰**을 봅니다. 각 토큰 임베딩을 이어붙여 MLP 에 넣는
= 진짜 **Bengio(2003) 입력층**이에요.

## v0.2.0 → v0.2.1, 무엇이 바뀌었나요?

> **문맥 길이 하나만 일반화.** 2토큰 고정 → 앞 N토큰(가변). N=2 로 두면 v0.2.0 과 동일.

| | v0.2.0 | v0.2.1 |
|---|---|---|
| 문맥 | 앞 2토큰 (고정) | **앞 N토큰** (`BLOCK_SIZE`) |
| 입력 | (B, 2) | (B, N) |
| 모델 | `Linear(2E, H)` | `Linear(N·E, H)` |
| 임베딩 concat | 2개 | **N개** (`emb(x).flatten(1)`) |

## 구조 (Bengio 2003)

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
        e = self.emb(x).flatten(1) # (B, N, E) → (B, N·E)  concat
        return self.fc2(torch.tanh(self.fc1(e)))
```

## 문장 맨 앞 — `<PAD>`

N토큰이 다 안 차는 문장 앞부분은 `<PAD>`(어휘 0번)로 채워요.
예) `ids=[a,b,c]`, N=3 → `(p,p,a)→b`, `(p,a,b)→c`, `(a,b,c)→<END>`.

## 구현 — 최소 override

`build_net`(BlockModel) · `make_pairs`(N토큰) · `_context_ids`(앞 N토큰 인덱스 목록) · `load`(저장된
모양에서 **N·E·H 복원**) 만 새로 정의. 임베딩·정규화·저장·대화·PPL 은 v0.2.0 상속.
새 하이퍼파라미터 **`BLOCK_SIZE`**(기본 3).

## 실행 방법

> ⚠️ **PyTorch** 필요. `pip install torch`

```bash
python3 1.train/train.py     # N토큰 문맥 학습 → 0.model/model.pt (+ vocab.json)
python3 2.test/test.py       # 학습/검증 PPL + 문장 완성 예시
```

기본 설정(N=3)의 검증 PPL 은 **3.37** 입니다.

> ⚠️ **이 숫자를 '문맥을 넓힌 결과'로 읽으면 안 돼요.** 이 버전은 `BLOCK_SIZE` 말고도
> `EMBED` 256→32 · `HIDDEN` 256→128 · `LR` · `INIT` · `LABEL_SMOOTHING` 까지 **5가지를 함께** 바꿨어요.
> 나머지를 고정하고 문맥만 2→3 으로 넓히면 검증 PPL 은 **2.9446 → 2.9182 로 오히려 좋아집니다.**
> 문맥 효과는 ±0.02(부호도 뒤집힘)인 반면 용량 효과는 0.43 — **3.37 의 원인은 문맥이 아니라 용량**이에요.
> 확인: 루트의 `ablation_block_size.py`.

- `BLOCK_SIZE=2` 는 v0.2.0 과 **완전히 동일**합니다(검증 2.9446 일치).
- `BLOCK_SIZE` 를 2·3·4 로 바꿔가며 비교해 보세요 — 단, E·H 를 함께 건드리면 원인을 말할 수 없어요.

## 임베딩 시대 진행 (v0.2.x)

- v0.2.0 임베딩 도입
- **v0.2.1 문맥 확장(`block_size`) ← 현재**
- v0.2.2 초기화·정규화 · v0.2.3 일반화·튜닝
- v0.2.4 임베딩 시각화 · v0.2.5 **3자 비교(캡스톤)**
