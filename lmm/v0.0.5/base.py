# -*- coding: utf-8 -*-
"""
base.py  (v0.0.5)  -  개수 -> 확률 + 온도(temperature) 로 다양하게 뽑기

[이 버전이 새로 더한 것]
  - choose(): 다음 토큰을 고를 때, '가장 많이 나온 것'만 고르지 않고
              개수를 확률로 바꿔 '뽑기' 를 합니다. 온도로 다양성을 조절해요.
              (부모의 그리디 choose 를 덮어씀)
                무게 = 개수 ^ (1 / 온도)   /   온도<=0 이면 그리디로.

나머지(백오프·문장끝·학습 등)는 전부 물려받아요.
"""

import os
import random
import importlib.util


def _inherit(prev_version):
    lmm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(lmm_dir, prev_version, "base.py")
    spec = importlib.util.spec_from_file_location("lmmbase_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NGramLM


class NGramLM(_inherit("v0.0.4")):
    def choose(self, counts, temperature):
        """개수를 '확률 + 온도' 로 바꿔서 다음 토큰을 뽑습니다."""
        candidates = list(counts.keys())
        nums = list(counts.values())
        if temperature <= 0.01:
            return candidates[nums.index(max(nums))]     # 온도 0 = 그리디
        weights = [c ** (1.0 / temperature) for c in nums]
        return random.choices(candidates, weights=weights, k=1)[0]
