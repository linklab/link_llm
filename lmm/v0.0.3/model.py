# -*- coding: utf-8 -*-
"""
v0.0.3 설정  -  문장 끝(<END>) 학습

v0.0.2 와 비교해 바뀐 점: base.py 가 <END> 를 추가 (prepare/is_end). 설정은 ORDERS=[2] 그대로.
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))

# 같은 폴더의 base.py 를 '파일 경로'로 불러옵니다. (dot 폴더라 일반 import 불가)
_spec = importlib.util.spec_from_file_location(
    "lmmbase_" + os.path.basename(_HERE).replace(".", "_"),
    os.path.join(_HERE, "base.py"),
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
NGramLM = _base.NGramLM


class Model(NGramLM):
    ORDERS = [2]


DATA_PATH = os.path.join(_HERE, "data", "data.txt")
MODEL_PATH = os.path.join(_HERE, "models", "model.json")
