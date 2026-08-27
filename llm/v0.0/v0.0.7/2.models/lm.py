# -*- coding: utf-8 -*-
"""
lm.py  (v0.0.7)  -  top-k / top-p 샘플링 (NGramLM 상속 + 설정)

[이 버전이 새로 더한 것]
  - generate(): top_k, top_p 를 받도록 확장 (부모 것을 감싸서 설정만 넘김)
  - choose(): 뽑기 전에 후보를 잘라냄
      top-k: 확률 상위 k개만 / top-p: 누적확률 p 까지만(nucleus) -> 재정규화 후 뽑기
백오프·온도·문장부호 토크나이저 등은 이전 버전에서 물려받아요.
"""

import os
import random
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


class NGramLM(_load_prev("v0.0.6")):
    def generate(self, start_text, temperature=0.0, top_k=0, top_p=1.0):
        """top_k, top_p 설정을 담아두고, 나머지 생성은 부모의 generate 에 맡깁니다."""
        self._top_k = top_k
        self._top_p = top_p
        return super().generate(start_text, temperature)

    def choose(self, counts, temperature):
        """온도로 무게를 만든 뒤, top-k / top-p 로 후보를 자르고, 남은 후보에서 뽑습니다."""
        candidates = list(counts.keys())
        nums = list(counts.values())

        if temperature <= 0.01:
            return candidates[nums.index(max(nums))]

        weights = [c ** (1.0 / temperature) for c in nums]
        total = sum(weights)
        probs = [w / total for w in weights]

        order = sorted(range(len(candidates)), key=lambda i: probs[i], reverse=True)

        top_k = getattr(self, "_top_k", 0)
        if top_k and top_k > 0:
            order = order[:top_k]

        top_p = getattr(self, "_top_p", 1.0)
        if top_p < 1.0:
            kept, cumulative = [], 0.0
            for i in order:
                kept.append(i)
                cumulative += probs[i]
                if cumulative >= top_p:
                    break
            order = kept

        kept_candidates = [candidates[i] for i in order]
        kept_weights = [probs[i] for i in order]
        return random.choices(kept_candidates, weights=kept_weights, k=1)[0]


class Model(NGramLM):
    ORDERS = [1, 2]


DATA_PATH = os.path.join(_DATA_DIR, "data.txt")
MODEL_PATH = os.path.join(_HERE, "model.json")
