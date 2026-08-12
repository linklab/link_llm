# -*- coding: utf-8 -*-
"""
v0.0.6 설정  -  문장부호 토크나이저

base.py 가 tokenize/detokenize 를 문장부호 분리 방식으로 덮어씀. 설정(ORDERS)은 v0.0.5 와 같음.
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
