# 2.models 폴더 (v0.1.0)

- `lm.py` : 이 버전 코드. v0.0.9 를 상속해 **확률 엔진만 신경망으로 교체**했어요.
- `model.json` : 학습 결과 (`3.train/train.py` 실행 시 자동 생성). **torch 환경에서 학습해야 생겨요.**

## 이 버전 모델의 특징

- **방식:** 개수 세기 → **신경망 학습**(경사하강). 미분은 PyTorch **autograd**(`loss.backward()`)가 담당.
- **구조:** 가중치 `W` (어휘수 V × V). `W[i]` 가 "앞 토큰 i 다음 토큰들의 점수(logits)" 행이에요.
  - 순전파: `logits = one-hot(prev) @ W == W[prev]` → `softmax` → 확률
  - 손실: `F.cross_entropy` (softmax + NLL 을 수치적으로 안정하게 한 번에)
  - 갱신: `W -= LR * W.grad` (**수동** 경사하강 — 옵티마이저는 v0.1.2에서)
- **문맥 길이:** 앞 1토큰 (bigram)
- **인터페이스 재사용(v0.0.x 그대로):** 토크나이저(`punct`) · `<END>` · 대화(`chat`) · 온도/top-k·top-p 샘플링 · 퍼플렉서티.
  → 딱 하나 `token_prob`/`next_token` 이 "개수 비율" 대신 **신경망 softmax 확률**을 돌려주도록만 바꿨어요.

## 클래스 이름

- 신경망 시대라 클래스명은 **`NeuralLM`** 을 씁니다. (v0.0.x 의 `NGramLM` 과 대비)
- 다만 `web_service` 와 상속 사슬이 `module.NGramLM` 을 찾으므로, 파일 끝에서
  `NGramLM = NeuralLM` 으로 **같은 클래스를 두 이름으로** 노출해 기존 인프라와 호환시켜요.

## model.json 형식 (v0.0.x 와 다름)

```json
{
  "type": "neural_bigram",
  "tokenizer": "punct",
  "vocab": ["<사용자>", "안녕", "<봇>", "...", "<END>"],
  "W": [[...], [...], "... (V x V 실수)"]
}
```

- 개수 표(`tables`) 대신 **어휘(`vocab`) + 가중치(`W`)** 를 저장해요.
- 불러올 때 `W` 를 다시 텐서로 만들어, `softmax(W[prev])` 로 다음 토큰 확률을 계산합니다.

## 왜 PPL 로 v0.0.9 와 비교하나

`perplexity()` 는 v0.0.9 것을 **그대로** 물려받고, `token_prob` 만 신경망 버전으로 바꿨어요.
그래서 **똑같은 자(尺)** 로 재게 되어, "학습이 개수 비율을 잘 재현했는지"를 공정하게 볼 수 있어요.
(본격적인 나란히 비교는 캡스톤 **v0.1.4**.)
