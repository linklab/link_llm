# -*- coding: utf-8 -*-
"""
base.py  (v0.0.7)  -  top-k / top-p 샘플링

[이 버전이 새로 더한 것]
  - generate(): top_k, top_p 를 받도록 확장 (부모 것을 살짝 감싸서 설정만 넘김)
  - choose(): 다음 토큰을 뽑기 '전에' 후보를 잘라내는 단계를 추가  (부모 것을 덮어씀)
      1) top-k: 확률이 높은 '상위 k개' 후보만 남김
      2) top-p: 확률을 높은 순으로 더해가며 '누적 p' 까지의 후보만 남김 (nucleus)
      3) 남은 후보만 다시 정규화해서 뽑기

[왜 필요한가]
온도만으로는 '엉뚱한 꼬리 후보'가 가끔 뽑혀요. top-k/top-p 는 그 꼬리를 잘라
안정성과 다양성의 균형을 맞춥니다. 이 방식은 '확률 분포' 에만 작용하기 때문에,
나중에 신경망(softmax 분포)으로 바뀌어도 '그대로' 재사용할 수 있어요.

(백오프·온도·문장부호 토크나이저 등은 이전 버전에서 그대로 물려받아요.)
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


class NGramLM(_inherit("v0.0.6")):
    # 기본값: 자르기 없음 (top_k=0 -> 끄기, top_p=1.0 -> 끄기)
    def generate(self, start_text, temperature=0.0, top_k=0, top_p=1.0):
        """top_k, top_p 설정을 담아두고, 나머지 생성은 부모(v0.0.1)의 generate 에 맡깁니다."""
        self._top_k = top_k
        self._top_p = top_p
        return super().generate(start_text, temperature)

    def choose(self, counts, temperature):
        """
        온도로 무게를 만든 뒤, top-k / top-p 로 후보를 자르고, 남은 후보에서 뽑습니다.
        (온도가 0 에 가까우면 그리디 — 자르기는 의미가 없어 생략)
        """
        candidates = list(counts.keys())
        nums = list(counts.values())

        if temperature <= 0.01:
            return candidates[nums.index(max(nums))]

        # 1) 온도 적용 -> 확률로 정규화
        weights = [c ** (1.0 / temperature) for c in nums]
        total = sum(weights)
        probs = [w / total for w in weights]

        # 2) 확률 높은 순으로 후보 인덱스 정렬
        order = sorted(range(len(candidates)), key=lambda i: probs[i], reverse=True)

        # 3) top-k: 상위 k개만 남김 (0 이면 끄기)
        top_k = getattr(self, "_top_k", 0)
        if top_k and top_k > 0:
            order = order[:top_k]

        # 4) top-p: 누적확률 top_p 까지만 남김 (1.0 이면 끄기). 최소 1개는 보장.
        top_p = getattr(self, "_top_p", 1.0)
        if top_p < 1.0:
            kept, cumulative = [], 0.0
            for i in order:
                kept.append(i)
                cumulative += probs[i]
                if cumulative >= top_p:
                    break
            order = kept

        # 5) 남은 후보만 다시 정규화해서 확률에 비례해 뽑기
        kept_candidates = [candidates[i] for i in order]
        kept_weights = [probs[i] for i in order]
        return random.choices(kept_candidates, weights=kept_weights, k=1)[0]
