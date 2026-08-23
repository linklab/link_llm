# -*- coding: utf-8 -*-
"""
lm.py  (v0.1.5)  -  기준선 대결 (캡스톤)  (NeuralLM 상속)

[이 버전의 '새 개념' 은 모델이 아니라 '비교' 예요]
  모델·학습은 v0.1.4 를 **그대로** 물려받아요 (2토큰 문맥 신경망 + 정규화·초기화).
  v0.1.5 가 더하는 건 —
    **여러 버전을 '같은 자(퍼플렉서티)'로 나란히 재서 비교(대결)** 하는 것.
  개수 세기 시대의 완성본 v0.0.9(카운트 bigram) vs 신경망 bigram(v0.1.x),
  같은 학습/검증 데이터로 PPL 을 재서 **"신경망이 정말 이겼나?"** 를 확인합니다.

[공정한 비교인 이유]
  모든 버전이 같은 토크나이저(punct)·같은 prepare(<END>)·같은 perplexity() 공식을 쓰고,
  딱 하나 '다음 토큰 확률을 어디서 얻느냐'(개수 표 ↔ 신경망 W)만 달라요.
  그래서 같은 문장에 PPL 을 재면 사과 대 사과 비교가 됩니다.

[사용]  4.test/test.py 가 load_version_model()/compare() 로 대결표를 출력해요.

[주의]  신경망 버전 모델을 부르려면 PyTorch 가 필요해요 (v0.0.9 카운트 모델은 표준 라이브러리만).
"""

import os
import importlib.util


_HERE = os.path.dirname(os.path.abspath(__file__))
_VERSION_DIR = os.path.dirname(_HERE)


def _load_prev_module(prev_version):
    """이전 버전 lm.py 모듈을 통째로 불러와요 (NGramLM 계보 재사용)."""
    group = prev_version.rsplit(".", 1)[0]                    # "v0.1.4" -> "v0.1" (마이너 그룹)
    llm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../llm 루트
    path = os.path.join(llm_dir, group, prev_version, "2.models", "lm.py")
    spec = importlib.util.spec_from_file_location("llmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# v0.1.4 를 그대로 물려받아요 (2토큰 문맥 신경망·정규화·초기화 동일).
_prev = _load_prev_module("v0.1.4")


class NeuralLM(_prev.NGramLM):
    pass          # 모델·학습은 v0.1.4(2토큰 문맥) 그대로. 이 버전의 새 개념은 아래 '비교' 도구예요.


# web_service / 상속 사슬이 module.NGramLM 을 찾으므로 노출.
NGramLM = NeuralLM


class Model(NeuralLM):
    pass


# ---------- 캡스톤: 여러 버전을 같은 데이터로 PPL 비교 ----------
def load_version_model(version):
    """
    다른 버전의 학습된 모델(model.json)을 그 버전 클래스로 불러와요.
    예) load_version_model("v0.0.9")  -> 카운트 bigram
        load_version_model("v0.1.5")  -> 이 2토큰 문맥 신경망
    (신경망 버전은 torch 가 필요하고, 아직 학습 안 됐으면 예외가 나요.)
    """
    group = version.rsplit(".", 1)[0]
    llm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))
    vdir = os.path.join(llm_dir, group, version)
    spec = importlib.util.spec_from_file_location(
        "llm_cmp_" + version.replace(".", "_"), os.path.join(vdir, "2.models", "lm.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # 신경망(v0.1.x) 은 model.pt(+vocab.json), 카운트(v0.0.x) 는 model.json 을 로드해요.
    mdir = os.path.join(vdir, "2.models")
    pt = os.path.join(mdir, "model.pt")
    model_file = pt if os.path.exists(pt) else os.path.join(mdir, "model.json")
    return module.NGramLM().load(model_file)


def compare(train_sents, valid_sents, versions):
    """
    각 버전 모델을 같은 문장으로 재요 — **두 가지 자**로 봅니다.
      · PPL      : 확률을 얼마나 잘 배분했나 (v0.0.9 perplexity)
      · 정확도    : 1등을 얼마나 맞혔나 + 정답이 후보에 있기라도 한가 (v0.0.9 accuracy)
    두 지표는 순위가 뒤바뀔 수 있어요 — 그게 '지표를 하나만 보면 안 되는' 이유예요.

    돌려주는 것: (rows, skipped)
      rows    = [(version, kind, ppl_train, ppl_valid, top1, coverage), ...]   # 불러온 것
      skipped = [(version, 이유), ...]                                          # 못 부른 것
    """
    rows, skipped = [], []
    for v in versions:
        try:
            lm = load_version_model(v)
        except SystemExit:                       # 신경망인데 torch 가 없음
            skipped.append((v, "PyTorch 필요(미설치)"))
            continue
        except FileNotFoundError:                # 아직 학습 안 됨(model.json 없음)
            skipped.append((v, "model.json 없음 — 먼저 그 버전 train.py 실행"))
            continue
        kind = "카운트" if v.startswith("v0.0") else "신경망"
        acc = lm.accuracy(valid_sents)           # 정확도는 '처음 보는' 검증 데이터로만
        rows.append((v, kind, lm.perplexity(train_sents), lm.perplexity(valid_sents),
                     acc["top1"], acc["coverage"]))
    return rows, skipped


DATA_PATH = os.path.join(_VERSION_DIR, "1.data", "data.txt")      # 학습용
VALID_PATH = os.path.join(_VERSION_DIR, "1.data", "valid.txt")    # 검증용
MODEL_PATH = os.path.join(_HERE, "model.pt")                     # 가중치 (PyTorch 표준)
VOCAB_PATH = os.path.join(_HERE, "vocab.json")                   # 어휘
