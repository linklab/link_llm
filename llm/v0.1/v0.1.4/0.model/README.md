# 0.model 폴더 (v0.1.4)

- `lm.py` : 이 버전 코드. v0.1.3 을 상속해 **문맥을 1토큰 → 2토큰**으로 넓혔어요.
- `model.json` : 학습 결과 (`1.train/train.py` 실행 시 생성). **torch 환경에서 학습해야 생겨요.**

## 이 버전의 목표 — 카운트와 '같은 문맥'으로 공정한 대결

v0.0.9(카운트)는 앞 2토큰(+백오프)을 쓰는데 신경망은 1토큰만 써서 불리했어요.
v0.1.4 는 신경망도 앞 2토큰을 보게 해 문맥 길이를 맞춥니다. 그러면 softmax 의 부드러움
(카운트의 FLOOR 절벽 회피)이 살아나 신경망이 이길 여지가 생겨요.

## 무엇을 바꿨나 (v0.1.3 → v0.1.4)

| # | 항목 | v0.1.3 | v0.1.4 |
|---|------|--------|--------|
| ① | 모델 | `BigramModel` `nn.Linear(V, V)` | `ContextModel` `nn.Linear(2V, V)` |
| ② | forward | one-hot(prev) | `cat[one-hot(prev2), one-hot(prev1)]` |
| ③ | 데이터 | `(prev) → next` | `(prev2, prev1) → next` (`make_pairs` 재정의) |
| ④ | 어휘 | 그대로 | 맨 앞에 `<PAD>` 추가(0번) |
| ⑤ | 저장 | `W` (V×V) | `W2` (V×2V), `type="neural_context2"` |

## logits 를 어떻게 얻나

one-hot 두 개를 이어붙였으므로(prev2 는 앞쪽 V칸, prev1 은 뒤쪽 V칸):

```python
logits[j] = W2[j, prev2] + W2[j, V + prev1]     # _probs_for2()
probs = softmax(logits)
```

## 그대로인 것 (상속)

- **정규화·초기화** (`make_optimizer` weight_decay · `init_model`) — v0.1.3 그대로.
- **Dataset/DataLoader** (`build_dataloader`) — 입력이 `(2,)` 벡터라도 그대로 배치돼요.
- **torch.optim · 대화(chat) · 샘플링 · 퍼플렉서티** — 전부 상속.
- 하이퍼파라미터는 `1.train/train.py` 에서 설정.

## 평가 포인트

`2.test/test.py` 의 **검증 PPL 이 v0.0.9(2.96) 아래**면 "같은 문맥에서 신경망 승".
1토큰 신경망(v0.1.0~3, 4점대)보다 문맥이 넓어 **2.90** 까지 내려가고, 카운트(2.96)도 넘어서요 — **신경망 승** ✅
(격차 0.06 · 파라미터는 99배. 조기 종료가 **검증 PPL 최저 에폭**의 가중치를 저장해 준 덕입니다.)
