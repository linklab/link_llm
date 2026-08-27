# -*- coding: utf-8 -*-
"""
lm.py  (v0.0.4)  -  백오프 (NGramLM 상속 + 설정)

[이 버전이 새로 더한 것]
  - next_token() / can_continue(): 긴 문맥(2단어)이 없으면 짧은 문맥(1단어)으로 줄여 재시도(백오프)
백오프가 동작하려면 표1·표2 가 둘 다 필요하므로 ORDERS = [1, 2] 로 둡니다.
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


class NGramLM(_load_prev("v0.0.3")):
    def next_token(self, recent, temperature):
        """백오프: 긴 문맥부터 찾고, 없으면 짧은 문맥으로 줄여서 다시 찾습니다."""
        for n in self._orders_desc():          # 예) 2 -> 1
            if len(recent) >= n:
                counts = self._tables()[str(n)].get(" ".join(recent[-n:]))
                if counts:
                    return self.choose(counts, temperature)
        return None

    def can_continue(self, recent):
        for n in self._orders_desc():
            if len(recent) >= n and self._tables()[str(n)].get(" ".join(recent[-n:])):
                return True
        return False


class Model(NGramLM):
    ORDERS = [1, 2]          # 표1 + 표2 -> 백오프 가능


DATA_PATH = os.path.join(_DATA_DIR, "data.txt")
MODEL_PATH = os.path.join(_HERE, "model.json")
