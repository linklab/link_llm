# -*- coding: utf-8 -*-
"""
v0.0.4 설정  -  백오프

base.py 가 백오프(next_token/can_continue)를 추가했고,
그게 동작하려면 표1·표2 가 둘 다 필요하므로 ORDERS = [1, 2] 로 둡니다.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from base import NGramLM


class Model(NGramLM):
    ORDERS = [1, 2]          # 표1 + 표2 -> 백오프 가능


DATA_PATH = os.path.join(_HERE, "train", "data.txt")
MODEL_PATH = os.path.join(_HERE, "models", "model.json")
