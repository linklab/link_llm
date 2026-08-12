# -*- coding: utf-8 -*-
"""
v0.0.3 설정  -  문장 끝(<END>) 학습

v0.0.2 와 비교해 바뀐 점: base.py 가 <END> 를 추가 (prepare/is_end). 설정은 ORDERS=[2] 그대로.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from base import NGramLM


class Model(NGramLM):
    ORDERS = [2]


DATA_PATH = os.path.join(_HERE, "train", "data.txt")
MODEL_PATH = os.path.join(_HERE, "models", "model.json")
