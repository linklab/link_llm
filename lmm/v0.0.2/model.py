# -*- coding: utf-8 -*-
"""
v0.0.2 설정  -  앞 '단어 2개' 로 예측

v0.0.1 과 비교해 바뀐 점: ORDERS = [1] -> [2]  (base.py 는 새 코드 없이 그대로!)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from base import NGramLM


class Model(NGramLM):
    ORDERS = [2]              # 앞 2단어를 봄


DATA_PATH = os.path.join(_HERE, "train", "data.txt")
MODEL_PATH = os.path.join(_HERE, "models", "model.json")
