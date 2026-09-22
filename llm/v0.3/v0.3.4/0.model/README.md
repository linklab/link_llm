# 모델 파일

- `network.py`: 독립적인 N개 Pre-LN 블록, 최종 LayerNorm, 깊이를 반영한 residual projection 초기화.
- `lm.py`: v0.3.3의 공통 학습·평가·생성을 재사용하며 층 수를 포함한 설정을 저장/복원.
- `model.pt`, `vocab.json`: 학습 명령으로 생성하는 로컬 산출물. Git 제외.
- `training_report.json`: 학습 설정·데이터 해시·검증 결과. 추론 체크포인트이며 학습 재개 기능은 아닙니다.

상세 설명과 실행 방법은 상위 README를 참고하세요.
