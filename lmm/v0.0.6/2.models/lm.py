# -*- coding: utf-8 -*-
"""
lm.py  (v0.0.6)  -  문장부호 토크나이저 (NGramLM 상속 + 설정)

[이 버전이 새로 더한 것]
  - tokenize()/detokenize(): 문장부호(. , ! ?)를 따로 떼어 하나의 토큰으로
  - tokenizer_name(): 저장 형식에 "punct" 로 기록
백오프·온도·문장끝 등은 이전 버전에서 물려받아요.
"""

import os
import re
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERSION_DIR = os.path.dirname(_HERE)


def _load_prev(prev_version):
    lmm_dir = os.path.dirname(_VERSION_DIR)
    path = os.path.join(lmm_dir, prev_version, "2.models", "lm.py")
    spec = importlib.util.spec_from_file_location("lmmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NGramLM


class NGramLM(_load_prev("v0.0.5")):
    PUNCT = {".", ",", "!", "?"}

    def tokenize(self, text):
        """문장부호를 따로 떼어 토큰으로 나눕니다."""
        return re.findall(r"[.,!?]|[^\s.,!?]+", text)

    def detokenize(self, tokens):
        """토큰을 문장으로 합칩니다. 문장부호는 앞 단어에 딱 붙여요."""
        text = ""
        for t in tokens:
            text += t if t in self.PUNCT else ((" " if text else "") + t)
        return text

    def tokenizer_name(self):
        return "punct"


class Model(NGramLM):
    ORDERS = [1, 2]


DATA_PATH = os.path.join(_VERSION_DIR, "1.data", "data.txt")
MODEL_PATH = os.path.join(_HERE, "model.json")
