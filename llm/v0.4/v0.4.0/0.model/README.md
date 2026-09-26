# 바이트 BPE 모델

- `tokenizer.py`: UTF-8 바이트 기반 BPE 학습·encode/decode·직렬화. 외부 토크나이저 라이브러리 불필요.
- `lm.py`: v0.3.4 네트워크 상속, 특수 출력 마스크, BOS/EOS, BPE 어휘, 원문 바이트 NLL 평가.
- `tokenizer.json`, `config.json`: 병합 규칙과 추론 설정을 읽을 수 있는 산출물.
- `model.pt`: 토크나이저·구조·평가 계약을 포함한 추론 체크포인트. Git 제외.
- `vocab.json`: 사람이 확인하는 체크포인트 메타데이터. Git 제외.
- `training_report.json`: 데이터/문서 해시·학습 설정·에폭별 로그·토큰 PPL·바이트당 NLL.

토큰 PPL은 기존 punct+josa PPL과 직접 비교하지 않습니다. 자세한 기준은 상위 README를 참고하세요.
