# v0.4.1 문서 코퍼스

이전 버전의 `data/pretrain/train.txt`·`valid.txt`를 덮어쓰지 않는 별도 데이터셋입니다. AI의 도움으로 이 저장소에 작성한 12개 주제별 문단 템플릿을 이름·시간으로 변형한 **교육용 합성 문서 480개**입니다. 외부 웹 문서나 다운로드한 코퍼스를 포함하지 않습니다. 사용 조건은 저장소 [MIT LICENSE](../../../LICENSE)를 따르며 출처·사용 조건은 `manifest.json`에 기록합니다.

## 문서 형식

JSONL 한 레코드가 문서 하나입니다. 문서 안의 줄바꿈은 JSON 문자열에 `\n`으로 저장하므로 여러 문단을 하나의 문서로 보존합니다. LF로 나눈 한 줄을 문서로 쓰던 구버전과 다릅니다.

```json
{"id":"garden-000","text":"첫 문단\n\n다음 문단","source":"authored-v041","topic":"garden","template_id":"garden-journal-v1"}
```

- 입력 필수 필드: `id`, `text`, `source`, `topic`, `template_id`. 빈 문자열 문서도 지원합니다.
- `source`는 출처 사전의 ID이며, 사전에는 `uri`, `license`, `usage`, `description`이 필요합니다. 사용자가 제출한 외부 자료의 권한을 자동 판정하는 기능은 아닙니다.
- 출력에는 `text_sha256`, `duplicate_key`, `group_id`를 추가합니다.
- 원문은 strip·Unicode 정규화 없이 저장합니다. U+2028·U+0085·CR·앞뒤 공백도 보존합니다.

## 중복과 분할

1. 중복 비교 키에만 **NFC + 공백 접기**를 적용합니다. 따라서 `가`/`가`, `a b`/`a  b`, 마지막 줄바꿈만 다른 문서도 중복으로 취급합니다. 보존하는 대표 원문은 ID 사전순으로 첫 문서입니다.
2. 같은 중복 키, 같은 topic, 같은 template_id를 연결해 그룹을 만듭니다. **중복을 제거하기 전에** 연결하므로 제거된 문서가 가지고 있던 그룹 관계도 유지됩니다.
3. 그룹을 시드 1234로 섞어 학습/검증/테스트에 나눕니다. 그룹을 쪼개서 목표 문서 수를 맞추지 않습니다. 기본 요청 비율은 그룹 기준 70/15/15이며, 12그룹을 반올림해 8/2/2그룹이 됩니다.
4. 독립 그룹이 3개보다 적으면 오류를 냅니다. 누수가 생기는 문서 무작위 분할로 자동 전환하지 않습니다.
5. 로더가 파일·원문 해시, 중복 문서, ID, 지정된 그룹의 split 간 교집합을 다시 검사합니다.

| split | 문서 수 | 주제·템플릿 그룹 | 원문 바이트 |
|---|---:|---:|---:|
| train | 320 | 8 | 157,734 |
| valid | 80 | 2: garden, library | 38,356 |
| test | 80 | 2: market, running | 39,416 |

원본 480개에서 이번 실행의 중복 제거 수는 0개입니다. 중복 제거 기능은 별도의 중복·분해형 Unicode·공백 변형 테스트로 검증합니다. 이름·시간만 바꾼 유사 문서는 같은 템플릿 그룹 안에 남습니다. 의미적 유사 중복이나 잘못 지정된 주제·템플릿을 자동 추론하지 않습니다. 분할은 **제공된 메타데이터 기준**이며 자연어 표현이 전혀 겹치지 않는다는 뜻은 아닙니다.

## 재생성

저장소 루트에서 표준 Python만으로 실행합니다.

```bash
python data/pretrain/v0.4.1/generate.py
# 별도 경로에서 동일한 JSONL/manifest 재현
python data/pretrain/v0.4.1/generate.py --output-dir /tmp/v041-corpus
```

`manifest.json`은 생성기·파이프라인·라이선스 파일의 SHA-256, 시드, 출처, 제거 내역, split별 파일 해시·문서 해시·바이트 수를 기록합니다. 생성기 코드가 바뀌면 같은 텍스트라도 manifest 해시가 바뀔 수 있습니다. 모델 체크포인트는 학습에 사용한 manifest 해시와 연결됩니다.

외부 JSONL은 출처 사전을 함께 제공해 처리합니다.

```bash
python data/pretrain/v0.4.1/corpus.py --input /path/documents.jsonl \
  --sources /path/sources.json --output-dir /tmp/my-corpus --group-by both
```

`--group-by both`가 기본이며 `topic`, `template_id`, `document`도 지원합니다. `document`는 중복만 격리하며 주제·템플릿 분리를 보장하지 않습니다. 모든 모드는 실제 선택한 정책을 manifest에 남깁니다. `sources.json`의 예:

```json
{"my-source":{"uri":"repo:my-documents","license":"사용자가 확인한 사용 조건","usage":"허용된 목적과 재배포 조건","description":"자료 출처 설명"}}
```

테스트 split은 데이터 무결성 검사에는 읽지만, 토크나이저 학습·신경망 학습·조기종료에는 쓰지 않습니다. 모델 선택이 끝난 뒤 전용 평가 명령의 `--split test`로만 명시적으로 채점합니다. 별도 출처의 대규모 벤치마크나 장거리 추론 평가를 대체하는 데이터는 아닙니다.
