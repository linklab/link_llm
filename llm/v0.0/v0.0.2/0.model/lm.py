# -*- coding: utf-8 -*-
"""
lm.py  (v0.0.2)  -  앞 '단어 2개' 로 예측 (NGramLM 상속 + 설정)

이 버전은 v0.0.1 의 NGramLM 을 그대로 물려받고, 설정만 ORDERS=[2] 로 바꿉니다.
(어떤 업그레이드는 새 코드 없이 '설정' 만으로 되기도 해요.)
"""

import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))     # 이 버전의 models 폴더
_VERSION_DIR = os.path.dirname(_HERE)                  # 이 버전 폴더 (v0.0.2)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_VERSION_DIR)))  # 저장소 루트
_DATA_DIR = os.path.join(_ROOT, "data")                                  # 모든 버전 공용 데이터


def _load_prev(prev_version):
    """이전 버전의 models/lm.py 를 불러와 그 NGramLM 을 돌려줍니다.
    (폴더 이름에 '.' 이 있어 보통의 import 가 안 되므로 파일 경로로 불러와요.)"""
    group = prev_version.rsplit(".", 1)[0]                    # "v0.0.9" -> "v0.0" (마이너 그룹)
    llm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../llm 루트
    path = os.path.join(llm_dir, group, prev_version, "0.model", "lm.py")
    spec = importlib.util.spec_from_file_location("llmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NGramLM


class NGramLM(_load_prev("v0.0.1")):
    pass   # v0.0.1 과 똑같은 코드. 달라지는 건 아래 ORDERS = [2] 뿐!


class Model(NGramLM):
    ORDERS = [2]              # 앞 2단어를 봄


DATA_PATH = os.path.join(_DATA_DIR, "pretrain", "train.txt")
MODEL_PATH = os.path.join(_HERE, "model.json")
