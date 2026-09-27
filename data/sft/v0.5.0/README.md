# SFT v0.5.0 데이터

48개 대화 후보와 출처·분할·검수 기록을 관리합니다. **AI 작성 후보이며 사람 승인 데이터는 아직 없습니다.**

`examples.jsonl`의 `messages`는 원문이고, `draft/`는 자동 검사를 통과한 후보 분할입니다. `reviews.jsonl`의 실제 사람 검수가 완료된 예제만 `--mode reviewed`로 내보낼 수 있습니다.

[구조·예시·실행·검수 방법](../../../llm/v0.5/v0.5.0/README.md)

```bash
python llm/v0.5/v0.5.0/1.train/prepare.py
python llm/v0.5/v0.5.0/2.test/test.py
```
