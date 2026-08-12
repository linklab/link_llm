# -*- coding: utf-8 -*-
"""
base.py  (v0.0.3)  -  문장 끝(<END>) 도 학습해서 자연스럽게 멈추기

[이 버전이 새로 더한 것]
  - END 상수: 문장 끝을 나타내는 가짜 토큰 "<END>"
  - prepare(): 학습할 때 문장 끝에 <END> 를 붙임  (부모의 prepare 를 덮어씀)
  - is_end(): 문장을 만들다 <END> 가 나오면 멈추게 함  (부모의 is_end 를 덮어씀)

나머지(학습 루프·생성 루프 등)는 v0.0.2 -> v0.0.1 에서 그대로 물려받아요.
"""

import os
import importlib.util


def _inherit(prev_version):
    lmm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(lmm_dir, prev_version, "base.py")
    spec = importlib.util.spec_from_file_location("lmmbase_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NGramLM


class NGramLM(_inherit("v0.0.2")):
    END = "<END>"                       # 문장 끝 표시 (가짜 토큰)

    def prepare(self, tokens):
        """학습 전에 문장 끝에 <END> 를 붙입니다."""
        return tokens + [self.END]

    def is_end(self, token):
        """문장을 만들다 <END> 를 만나면 멈춥니다."""
        return token == self.END
