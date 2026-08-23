# -*- coding: utf-8 -*-
"""
lm.py  (v0.0.5)  -  개수 -> 확률 + 온도 (NGramLM 상속 + 설정)

[이 버전이 새로 더한 것]
  - choose(): '가장 많이 나온 것'만 고르지 않고, 개수를 확률로 바꿔 '뽑기'.
              온도(temperature)로 다양성 조절.  무게 = 개수 ^ (1 / 온도)
학습 설정(ORDERS)은 v0.0.4 와 같고, 온도는 생성할 때 정합니다.
"""

import os
import random
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERSION_DIR = os.path.dirname(_HERE)


def _load_prev(prev_version):
    group = prev_version.rsplit(".", 1)[0]                    # "v0.0.9" -> "v0.0" (마이너 그룹)
    llm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../llm 루트
    path = os.path.join(llm_dir, group, prev_version, "2.models", "lm.py")
    spec = importlib.util.spec_from_file_location("llmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NGramLM


class NGramLM(_load_prev("v0.0.4")):
    def choose(self, counts, temperature):
        """개수를 '확률 + 온도' 로 바꿔서 다음 토큰을 뽑습니다."""
        candidates = list(counts.keys())
        nums = list(counts.values())
        if temperature <= 0.01:
            return candidates[nums.index(max(nums))]     # 온도 0 = 그리디
        weights = [c ** (1.0 / temperature) for c in nums]
        return random.choices(candidates, weights=weights, k=1)[0]


class Model(NGramLM):
    ORDERS = [1, 2]


DATA_PATH = os.path.join(_VERSION_DIR, "1.data", "data.txt")
MODEL_PATH = os.path.join(_HERE, "model.json")
