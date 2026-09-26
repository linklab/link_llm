# v0.4.1 모델 산출물

- `lm.py`: v0.4.0 BPE 모델 상속, 문서별 attention·위치 초기화, 패킹 학습 루프.
- `packing.py`: 문서를 겹치지 않는 정답 청크로 나누고 순서대로 패킹·PAD 처리.
- `model.pt`: 가중치·어휘·BPE·모델 설정·패킹 정책·학습 corpus manifest 해시. Git 제외.
- `tokenizer.json`, `config.json`, `vocab.json`: 사람이 읽는 추론 설정. vocab.json은 Git 제외.
- `training_report.json`: 학습/검증 지표·에폭별 이력·패킹 집계·환경·학습 시간·체크포인트 SHA-256. 테스트 선택에 사용하지 않습니다.
- `valid_report.json`, `test_report.json`: 고정 체크포인트의 전체·긴 문서·주제별 평가와 생성 예시. `test_report.json`은 별도 명시적 평가 결과입니다.

체크포인트는 추론용이며 optimizer/RNG를 저장한 재개용 체크포인트가 아닙니다. 원문 바이트당 NLL의 정의는 v0.4.0과 같지만 코퍼스·토크나이저·학습 문맥이 달라 과거 수치와 직접 순위를 비교하지 않습니다.
