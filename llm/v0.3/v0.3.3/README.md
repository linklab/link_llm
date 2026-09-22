# v0.3.3 — Pre-LN Transformer 블록

## 한 줄 요약

> 어텐션과 FFN 앞에 LayerNorm을 두고, 각 계산 결과를 원래 표현에 더하는 단일 Transformer 블록을 만듭니다.

## 왜 필요한가?

- 어텐션으로 모은 문맥 정보를 FFN이 토큰별로 비선형 가공합니다.
- 잔차 연결은 입력 표현과 gradient가 변환 경로를 우회해 전달될 수 있게 합니다.
- Pre-LN은 어텐션과 FFN에 들어가기 전에 각 토큰의 특징 차원을 정규화합니다. 배치 전체 통계를 쓰는 BatchNorm과 다릅니다.

## 무엇이 바뀌었나?

- v0.3.2의 토큰+위치 임베딩과 멀티헤드 어텐션을 유지합니다.
- 서로 독립적인 LayerNorm 두 개와 잔차 경로 두 개를 추가했습니다.
- FFN: `Linear(64,256) → GELU → Linear(256,64)`. 같은 가중치를 모든 토큰 위치에 적용하며 미래 토큰을 섞지 않습니다.
- 각 잔차 분기에 dropout 기본 0.1을 적용합니다. 학습 시 활성화, 평가·생성·로드 후에는 비활성화됩니다.
- `--ffn-hidden`, `--dropout`을 CLI와 체크포인트 메타데이터에 추가했습니다. `--dropout 0`으로 끌 수 있습니다.

## 모델 내부 구조

```text
h = token_embedding + position_embedding
h = h + Dropout(MultiHeadAttention(LayerNorm1(h)))
h = h + Dropout(FFN(LayerNorm2(h)))
logits = vocabulary_head(h)
```

각 단계 후 PAD 표현은 0으로 유지합니다. **블록 1개**이며 다층 적층과 GPT-2식 초기화는 v0.3.4 범위입니다.
어텐션 마스크·정답 이동·문맥창·FLOOR는 기존과 동일합니다.
기본 FFN 폭 256·dropout 0.1 외에는 문맥 32·임베딩 64·4헤드·Adam 0.003·배치 64·최대 60에폭·patience 10·시드 1234를 유지합니다.

## 효과와 검증

기본 설정 CPU 실측: **168,521개 파라미터**, 검증 **PPL 3.1488**, **top-1 74.75%**.
검증 최저 epoch 8을 채택하고 18에폭에서 조기종료했습니다. 채점 10,197자리, coverage 95.53%입니다.

- 14개 CPU 테스트: v0.3.2의 12개 검증과 블록/모드 검증 2개.
- Pre-LN 순서와 두 잔차 덧셈을 직접 계산한 결과와 대조합니다.
- 어텐션 projection·두 LayerNorm·FFN 두 층으로 유한한 gradient가 전달되는지 확인합니다.
- 두 분기의 출력을 0으로 만들면 블록 출력이 입력과 정확히 같아 잔차 경로가 남는지 검사합니다.
- 학습 모드의 dropout, 평가 모드의 동일 입력 재현, 평가 후 원래 학습 모드 복원을 확인합니다.
- 기본값과 다른 FFN 폭·dropout의 저장/로드 및 CPU 장치 고정으로 정확한 logits 일치를 검사합니다.
- 실측 데이터 해시·설정·환경·결과는 `0.model/training_report.json`에 기록됩니다.

## 한계

- 단일 블록 교육용 모델이며 완성된 다층 GPT나 지시학습 어시스턴트는 아닙니다.
- LayerNorm·잔차·FFN·dropout을 함께 추가하므로 성능 차이를 한 기법만의 효과로 해석하지 않습니다.
- 검증 수치는 한 시드의 모델 선택 결과입니다. 별도 최종 테스트·반복 실험 없이 우위를 일반화하지 않습니다.
- 위치 표의 범위, OOV 처리, 슬라이딩 비용, 학습 재개 미지원은 이전 버전과 같습니다.

## 실행·코드

```bash
python llm/v0.3/v0.3.3/2.test/test_invariants.py
python llm/v0.3/v0.3.3/1.train/train.py --device cpu --ffn-hidden 256 --dropout 0.1
python llm/v0.3/v0.3.3/2.test/test.py --prompt "오늘은"
python web_service/server.py
```

PyTorch가 설치된 Python을 사용합니다. `network.py`는 v0.3.2의 문맥 변환을 확장하고,
`lm.py`는 신경망 생성과 FFN/dropout 메타데이터를 지정합니다. 데이터·학습·평가 코드는 이전 버전에서 상속합니다.
학습한 모델이 있는 컴퓨터에서는 웹앱에 v0.3.3이 표시됩니다. 가중치는 Git에 포함되지 않습니다.
