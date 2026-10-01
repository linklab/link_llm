# v0.5.1 — 학습과 추론이 공유하는 대화 포맷

v0.5.0의 대화 데이터를 **누가 말했는지, 어디서 응답을 시작하는지, 언제 끝나는지** 구분할 수 있는 토큰열로 바꿉니다. 학습용 정답과 추론용 입력이 같은 형식을 사용합니다.

이 버전은 포맷·데이터 배포 묶음과 입력 호환 모델 구성을 제공합니다. 모델 학습·가중치 저장·웹 채팅 연결은 아직 하지 않습니다. 응답 토큰만 학습하는 처리는 v0.5.2, 실제 SFT는 v0.5.3의 범위입니다.

## git pull 후 준비 완료 확인

```bash
git pull origin main
conda activate link_llm
# 원본·AI 검토 기록·토큰열·모델 설정·해시를 한 번에 확인 (표준 라이브러리)
python llm/v0.5/v0.5.1/1.train/release.py --check
# 실제 네트워크 구성과 전체 데이터 입력까지 확인 (PyTorch 필요)
python llm/v0.5/v0.5.1/2.test/test_release.py
```

`1.train/release/`에 내려받는 자료가 모두 들어 있습니다. 재생성은 `release.py`를 옵션 없이 실행합니다. 기존 내용과 다르면 덮어쓰지 않으므로 변경한 데이터는 `--output-dir`로 새 경로에 내보내세요.

| 배포 파일 | 내용 |
|---|---|
| `inputs/` | 대화 원본·출처·사람 검수 파일·48개 AI 검토 근거 |
| `prepared/` | 원본 분할·공통 템플릿·토큰열·메시지 위치 |
| `model_config.json` | 어휘 769·문맥 256·RoPE·폭64·4층·4헤드·FFN256·weight tying |
| `readiness.json` | 데이터/모델 구성 상태·원문 복원 결과·전체 파일 해시 |

**데이터 준비 완료(`preparation_complete=true`)**와 **사람 검수 후 학습 승인(`training_ready`)**은 별개입니다. AI가 48개 대화와 54개 응답을 읽고 검토했으며, 사람 승인으로 기록하지 않았습니다. 현재는 교육용 실험 자료입니다. 내용 검토 이력은 [품질 검토 기록](../../../../data/sft/v0.5.0/QUALITY_REVIEW.md)에 있습니다.

## 간단한 예시

```text
system: 짧게 답하세요.
user: 1 더하기 1은?
assistant: 2입니다.
```

학습할 때는 다음 순서로 저장합니다. 아래 기호는 실제로는 각각 하나의 제어 토큰 ID입니다.

```text
<BOS>
<SYSTEM>짧게 답하세요.<EOT>
<USER>1 더하기 1은?<EOT>
<ASSISTANT>2입니다.<EOT>
<EOS>
```

추론할 때는 정답을 빼고 assistant가 말할 위치까지만 제공합니다.

```text
<BOS><SYSTEM>짧게 답하세요.<EOT><USER>1 더하기 1은?<EOT><ASSISTANT>
```

이 입력은 학습용 토큰열에서 `2입니다.` 직전까지와 정확히 같습니다. 예시의 줄바꿈은 설명용이며, 템플릿이 본문에 공백이나 줄바꿈을 추가하지 않습니다.

여러 번 주고받는 대화도 같은 규칙입니다.

```text
<BOS><USER>내 이름은 민수야.<EOT><ASSISTANT>반가워요.<EOT>
<USER>내 이름이 뭐지?<EOT><ASSISTANT>민수입니다.<EOT><EOS>
```

## 토큰과 종료 규칙

기존 v0.4.6 토크나이저의 어휘와 BPE 병합 규칙을 그대로 사용하고, 마지막에 `<EOT>`만 추가합니다. SFT 데이터로 BPE를 다시 학습하지 않습니다.

| 구분 | 기본 토크나이저에서의 ID | 의미 |
|---|---:|---|
| PAD / EOS / BOS | 0 / 1 / 2 | 패딩 / 대화 전체 종료 / 대화 시작 |
| USER / ASSISTANT / SYSTEM | 3 / 4 / 5 | 발화자 구분 |
| 기존 바이트·BPE 토큰 | 6~767 | 본문 표현, 번호 유지 |
| EOT | 768 | 한 메시지 종료 |

기존 어휘는 768개, 대화용 어휘는 769개입니다. 다른 기본 토크나이저를 지정하면 EOT ID는 그 토크나이저의 어휘 수가 됩니다.

- 모든 메시지는 EOT로 끝나고, 완성된 대화 마지막에만 EOS를 붙입니다.
- 생성 입력은 user 메시지로 끝나야 하며, 마지막에 ASSISTANT를 열어 둡니다.
- 생성된 응답만 `decode_response()`에 전달합니다. EOT는 `turn_end`, EOS 또는 EOT+EOS는 `conversation_end`, 종료 토큰이 없으면 `incomplete`입니다.
- 첫 system 메시지는 선택 사항이며, 이후에는 user와 assistant가 번갈아 등장해야 합니다.
- 사용자가 본문에 `<ASSISTANT>` 또는 `<EOT>`라고 입력해도 일반 글자로 인코딩합니다. 본문 문자열이 발화자나 종료 토큰으로 바뀌지 않습니다.
- 한글·이모지·분해된 한글·연속 공백·줄바꿈을 그대로 복원합니다. 잘못된 UTF-8 응답을 숨기거나 삭제하지 않고 오류로 알립니다.

**기존 모델에 ID 768을 바로 입력하면 안 됩니다.** `0.model/model_setup.py`는 어휘 769개·문맥 256의 새 네트워크를 구성하며, EOT 입력·출력과 weight tying을 지원합니다. 48개 대화의 순전파와 EOT 기울기를 CPU에서 검증했습니다. 이는 무작위 초기화한 구성 검사입니다. 기존 체크포인트를 수정하거나 사전학습 가중치를 옮기지는 않으며, 가중치 이식·학습·생성 루프 연결은 v0.5.3에서 수행합니다.

## 파일과 API

| 파일 | 역할 |
|---|---|
| `0.model/chat_template.py` | 공통 포맷, 인코딩·복원, 템플릿 저장·검증 |
| `1.train/prepare.py` | 대화를 토큰열로 변환하고 원본·분할·검수 기록 보존 |
| `1.train/preview/` | 미검수 후보 48개의 확인용 변환 결과 |
| `2.test/test.py` | 전체 대화 복원과 각 응답의 입력 접두부 검증 |
| `2.test/test_invariants.py` | 경계·변조·검수·길이·CLI 테스트 |
| `2.test/verification_report.json` | 확인용 데이터 검증 결과 |

`ChatTemplate`의 주요 메서드:

| 메서드 | 입력 → 출력 |
|---|---|
| `encode_conversation(messages)` | assistant로 끝나는 대화 → `input_ids`, 메시지 위치, EOS 위치 |
| `encode_prompt(messages)` | user로 끝나는 대화 → `input_ids`, 메시지 위치, 응답 시작 위치 |
| `decode_conversation(ids)` | 완성된 토큰열 → 원래 메시지 목록 |
| `decode_response(ids)` | 새로 생성한 토큰열 → 본문·종료 이유 |
| `save(path)` / `load(path)` | BPE 설정을 포함한 템플릿 저장·복원 |

메시지 위치는 시작 포함·끝 제외입니다. `start`는 역할 토큰, `content_start:content_end`는 본문, `content_end`는 EOT, `end`는 EOT 다음 위치입니다. 손실 계산용 labels와 마스크는 아직 만들지 않습니다.

## 실행 방법

저장소 루트에서 실행합니다. 이 버전의 변환·검증 도구는 Python 표준 라이브러리만 사용합니다.

```bash
# 미검수 후보를 명시적으로 확인용으로 변환
python llm/v0.5/v0.5.1/1.train/prepare.py --allow-draft

# 원문 복원·학습/추론 접두부·응답 종료 검증
python llm/v0.5/v0.5.1/2.test/test.py --allow-draft

# 자동 테스트
python llm/v0.5/v0.5.1/2.test/test_invariants.py
python llm/v0.5/v0.5.0/2.test/test_invariants.py
```

`--allow-draft` 없이 미검수 데이터를 사용하면 거부합니다. 이 옵션은 검수 완료로 바꾸는 기능이 아닙니다. 사람이 검수하여 v0.5.0 도구로 만든 reviewed 데이터는 별도 경로로 지정합니다.

```bash
python llm/v0.5/v0.5.1/1.train/prepare.py \
  --dataset data/sft/v0.5.0/reviewed \
  --output-dir llm/v0.5/v0.5.1/1.train/reviewed

python llm/v0.5/v0.5.1/2.test/test.py \
  --dataset llm/v0.5/v0.5.1/1.train/reviewed
```

위 reviewed 경로는 검수 후 생성해야 합니다. 원본 파일은 `source/`에 그대로 복사하며, ID·분할·검수 상태를 유지합니다. 파일 해시뿐 아니라 원본을 다시 인코딩한 결과까지 비교합니다. 기존 출력과 내용이 다르면 덮어쓰지 않고 새 출력 경로를 요구합니다.

## 검증 결과와 다음 단계

- v0.5.1 포맷 테스트 **21개**, 배포·실제 모델 구성 테스트 **5개**, v0.5.0 회귀 테스트 **18개** 통과.
- 기존 BPE의 원문 복원·병합 결정성·손상 거부 테스트 **3개** 통과.
- 대화 **48개** 원문 완전 복원, assistant 응답 **54곳**에서 학습/추론 입력 접두부 및 응답 복원 일치.
- 저장한 템플릿 재사용, 본문 속 제어 토큰 문자열, 역할 순서 오류, 데이터·위치 정보 변조, 미검수 데이터 거부, 출력 보호 검증.

| 분할 | 대화 수 | 총 토큰 | 대화당 토큰 범위 | 64토큰 초과 |
|---|---:|---:|---:|---:|
| train | 16 | 2,441 | 118~208 | 16 |
| valid | 16 | 2,643 | 122~222 | 16 |
| test | 16 | 2,388 | 117~184 | 16 |

현재 데이터는 **AI 내용 검토 완료·사람 검수 전(`training_ready: false`)**입니다. 모든 대화가 기존 64토큰 문맥을 넘지만, 새 배포 설정의 256토큰 안에는 잘림 없이 들어갑니다. `--max-tokens 64`를 지정하면 저장 전에 오류로 알립니다. v0.5.2에서 응답 손실 마스킹과 향후 256토큰을 초과하는 데이터의 처리 정책을 구현합니다. 256토큰 네트워크 입력이 가능하다는 결과가 학습된 장거리 이해 능력을 뜻하지는 않습니다.

이 검증은 포맷의 정확성을 확인합니다. 자연스러운 답변이나 지시 이행 능력은 검수된 데이터로 실제 SFT를 수행한 뒤 별도로 평가해야 합니다.
