# v0.4.5 — 현대 구조를 하나씩 비교하기

**기준 모델에서 부품 하나만 바꾼 5개 후보를 같은 조건으로 학습합니다.** 네 기술을 한꺼번에 합치지 않아 무엇을 바꿨는지 분명하게 볼 수 있습니다.

| 후보 | 바뀐 부품 | 쉽게 이해하기 |
|---|---|---|
| baseline | 없음 | 학습형 위치·LayerNorm·GELU FFN·일반 멀티헤드 |
| rope | 위치 표현 | Q·K를 위치별로 회전해 두 토큰 사이의 거리를 반영 |
| rmsnorm | 정규화 | 평균을 빼지 않고 값의 제곱 평균 크기를 맞춤 |
| swiglu | FFN | 내용 경로에 SiLU 게이트를 곱해 전달량을 조절 |
| gqa | K·V 헤드 | Query 4개가 Key·Value 2개를 나누어 사용 |

RoPE는 학습형 위치 임베딩을 제거합니다. RMSNorm은 모든 LayerNorm을 대체합니다. SwiGLU의 중간 폭은 170(기존256의 약2/3)이며 세 행렬의 총 규모를 비슷하게 맞췄지만 파라미터가 정확히 같지는 않습니다. GQA는 두 Query 헤드마다 K·V 한 쌍을 공유합니다.

## 고정 비교 조건·결과

문맥64·폭64·4층·Query4헤드·가중치 공유·dropout0.1·배치128·4에폭·시드1234·CPU4스레드. 같은 train/valid, train-only BPE, AdamW·warmup/cosine·clip을 사용합니다. **test는 선택에 사용하지 않았습니다.** 시작 전에 valid NLL/byte 최저 후보를 고르는 규칙을 기록했습니다.

| 후보 | 파라미터 | 학습 정답 토큰 | valid PPL ↓ | NLL/byte ↓ | 학습+검증 초 | K/V KiB |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 253,056 | 194,720 | 190.287 | 2.1978 | 43.12 | 96 |
| rope | 248,960 | 194,720 | 150.858 | 2.1009 | 47.26 | 96 |
| rmsnorm | 252,480 | 194,720 | 185.815 | 2.1880 | 42.55 | 96 |
| swiglu | 251,264 | 194,720 | 192.869 | 2.2036 | 44.45 | 96 |
| gqa | 236,672 | 194,720 | 168.687 | 2.1474 | 40.14 | 48 |

선택 모델은 **rope**입니다. 최선 검증 에폭의 모델·토크나이저·재개 파일을 `0.model/`에 저장했습니다. 각 후보의 학습량은 같지만 파라미터 수와 계산량은 다릅니다. v0.4.3의 폭96·6층·8에폭 모델과 직접 대조해 특정 구조가 더 좋다고 결론 내리지 않습니다.

K/V 크기는 별도 구조 비용 실험의 48토큰 캐시입니다. CPU4스레드에서 같은 형태의 무작위 가중치로 prefill 포함33회 예측을 측정했습니다. 속도 원시는 비용 보고서에 있으며 언어 품질 측정은 아닙니다. GQA의 K/V 텐서 저장량은 baseline의 절반입니다. 전체 메모리가 절반이라는 뜻은 아닙니다.

## 검증·실행

**테스트 9개 통과.** 5개 구조 각각의 인과/PAD/문서 마스크, gradient 유한성, 독립 청크와 logits 일치, KV 캐시·문맥창 넘침, 저장 복원, dropout 중간 재개를 검증했습니다. RoPE의 벡터 길이·상대 위치 성질과 RMSNorm/SwiGLU 수식, GQA 헤드 수·캐시 크기도 확인했습니다.

```bash
python llm/v0.4/v0.4.5/2.test/test_invariants.py
# 5개 후보 학습 후 valid로만 선택; 기본 모델 산출물 갱신
python llm/v0.4/v0.4.5/1.train/compare.py
python llm/v0.4/v0.4.5/2.test/architecture_benchmark.py
python llm/v0.4/v0.4.5/2.test/test.py --split valid
# 후보 하나만 별도 실험
python llm/v0.4/v0.4.5/1.train/train.py --variant rope --output-dir /tmp/link-rope
```

[선언한 비교 규칙·모든 결과](0.model/architecture_report.json) · [구조별 비용](0.model/architecture_cost.json) · [선택 모델](0.model/training_report.json)

단일 시드·작은 합성 코퍼스의 교육용 비교입니다. 최신 구조라는 이유만으로 품질·속도 우위를 주장하지 않습니다. 여러 구조를 합친 모델이나 장비별 최적화는 이번 비교 범위에 포함하지 않습니다.
