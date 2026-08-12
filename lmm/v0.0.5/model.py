# -*- coding: utf-8 -*-
"""
v0.0.5 설정  -  개수 -> 확률 + 온도

base.py 가 choose(확률+온도)를 추가. 설정(ORDERS)은 v0.0.4 와 같고, 온도는 생성할 때 정합니다.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from base import NGramLM


class Model(NGramLM):
    ORDERS = [1, 2]


DATA_PATH = os.path.join(_HERE, "train", "data.txt")
MODEL_PATH = os.path.join(_HERE, "models", "model.json")
