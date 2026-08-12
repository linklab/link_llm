# -*- coding: utf-8 -*-
"""
base.py  (v0.0.6)  -  문장부호(. , ! ?)를 따로 떼는 토크나이저

[이 버전이 새로 더한 것]
  - tokenize(): 문장부호를 하나씩 따로 떼어냄  ("좋아!" -> ["좋아", "!"])
  - detokenize(): 문장을 합칠 때 문장부호를 앞 단어에 딱 붙임
  - tokenizer_name(): 저장 형식에 "punct" 로 기록
  (모두 부모의 것을 덮어씀)

나머지(백오프·온도·문장끝·학습 등)는 전부 물려받아요.
"""

import os
import re
import importlib.util


def _inherit(prev_version):
    lmm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(lmm_dir, prev_version, "base.py")
    spec = importlib.util.spec_from_file_location("lmmbase_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NGramLM


class NGramLM(_inherit("v0.0.5")):
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
