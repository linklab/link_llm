# -*- coding: utf-8 -*-
"""
lm.py  (v0.0.3)  -  문장 끝(<END>) 학습 (NGramLM 상속 + 설정)

[이 버전이 새로 더한 것]
  - END 상수 / prepare(): 학습할 때 문장 끝에 <END> 를 붙임
  - is_end(): 문장을 만들다 <END> 가 나오면 멈춤
나머지는 v0.0.2 -> v0.0.1 에서 물려받아요.
"""

import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERSION_DIR = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_VERSION_DIR)))  # 저장소 루트
_DATA_DIR = os.path.join(_ROOT, "data")                                  # 모든 버전 공용 데이터


def _load_prev(prev_version):
    group = prev_version.rsplit(".", 1)[0]                    # "v0.0.9" -> "v0.0" (마이너 그룹)
    llm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../llm 루트
    path = os.path.join(llm_dir, group, prev_version, "2.models", "lm.py")
    spec = importlib.util.spec_from_file_location("llmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NGramLM


class NGramLM(_load_prev("v0.0.2")):
    END = "<END>"                       # 문장 끝 표시 (가짜 토큰)

    def prepare(self, tokens):
        """학습 전에 문장 끝에 <END> 를 붙입니다."""
        return tokens + [self.END]

    def is_end(self, token):
        """문장을 만들다 <END> 를 만나면 멈춥니다."""
        return token == self.END


class Model(NGramLM):
    ORDERS = [2]


DATA_PATH = os.path.join(_DATA_DIR, "data.txt")
MODEL_PATH = os.path.join(_HERE, "model.json")
