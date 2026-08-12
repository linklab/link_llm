# -*- coding: utf-8 -*-
"""
v0.0.7 설정  -  top-k / top-p 샘플링

base.py 가 choose()에 top-k/top-p 자르기를 추가. 설정(ORDERS)은 v0.0.6 과 같음.
top_k, top_p 는 '생성할 때' 정하는 값이라 여기(설정)에는 없어요 (test/웹에서 지정).
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
    ORDERS = [1, 2]


DATA_PATH = os.path.join(_HERE, "data", "data.txt")
MODEL_PATH = os.path.join(_HERE, "models", "model.json")
