# -*- coding: utf-8 -*-
"""
v0.0.1 설정  -  가장 기본: 앞 '단어 1개' 로 예측

함수(코드)는 같은 폴더의 base.py 에 있어요.
이 파일은 v0.0.1 이 '무엇이 다른가'(설정)만 정합니다.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # 이 버전 폴더
sys.path.insert(0, _HERE)                              # 같은 폴더의 base.py 를 찾기 위해
from base import NGramLM


class Model(NGramLM):
    ORDERS = [1]              # 앞 1단어만 봄


DATA_PATH = os.path.join(_HERE, "train", "data.txt")
MODEL_PATH = os.path.join(_HERE, "models", "model.json")
