# v0.3.5 모델과 비교 산출물

- `lm.py`: v0.3.4를 상속하는 웹앱용 v0.3.5 GPT.
- `comparison.py`: 카운트/MLP 어댑터, 공통 확률·신규성 평가, 후보 재학습, 두 비교 트랙, JSON/Markdown 보고서.
- `comparison_report.json`, `comparison_report.md`: 학습 CLI가 생성하는 실측 보고서.
- `candidates/`: 후보별 로컬 모델. 각 `0.model/model.pt` 또는 `model.json`은 Git 제외.
- `model.pt`, `vocab.json`: 첫 지정 시드의 후보 내 최선 GPT. 기본 웹앱이 자동 검색합니다.

자세한 계약·한계·실행은 상위 README에 있습니다. 보고서는 독립 최종 테스트 결과가 아닙니다.
