# -*- coding: utf-8 -*-
"""
v0.0.1 설정  -  가장 기본: 앞 '단어 1개' 로 예측

함수(코드)는 같은 폴더의 base.py 에 있어요.
이 파일은 v0.0.1 이 '무엇이 다른가'(설정)만 정합니다.
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
    ORDERS = [1]              # 앞 1단어만 봄


DATA_PATH = os.path.join(_HERE, "data", "data.txt")
MODEL_PATH = os.path.join(_HERE, "models", "model.json")
