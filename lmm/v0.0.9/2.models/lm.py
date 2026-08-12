# -*- coding: utf-8 -*-
"""
lm.py  (v0.0.9)  -  퍼플렉서티(perplexity) 평가 + 데이터 정비 (NGramLM 상속)

[이 버전이 새로 더한 것]  "얼마나 잘 맞히나?" 를 숫자로 재기
  - token_prob(): 지금 문맥 다음에 특정 토큰이 올 '확률' 을 구함 (백오프 사용)
  - perplexity(): 여러 문장에 대한 퍼플렉서티(PPL) 를 계산
      PPL = exp( 평균( -log p(각 토큰) ) )   →  낮을수록 '잘 맞힘'
      직관: "다음 토큰을 정할 때 평균 몇 개 중에서 헷갈리나". PPL=5 면 5개 중 하나 고르는 수준.

[데이터 정비]  1.data/ 를 학습용(data.txt)과 검증용(valid.txt)으로 나눴어요.
  - 학습 데이터: 모델이 외운 것 → PPL 이 낮음
  - 검증 데이터: 처음 보는 것 → PPL 이 더 높음  (이 차이가 '외우기 vs 일반화' 격차)

이건 개수 세기 시대의 마지막 정비예요. 나중에 신경망(v0.1.x~)이 '정말 더 나은지'
이 PPL 숫자로 비교할 기준선이 됩니다.

(대화 기능은 v0.0.8 그대로 물려받아요.)
"""

import os
import math
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


class NGramLM(_load_prev("v0.0.8")):
    FLOOR = 1e-4   # 처음 보는(학습에서 못 본) 토큰에 주는 아주 작은 확률

    def token_prob(self, recent, token):
        """
        지금 문맥(recent) 다음에 'token' 이 올 확률.
        백오프로 찾고(앞 2 -> 1), 표에 없으면 아주 작은 값(FLOOR)을 줍니다.
          p = 그 문맥에서 token 이 나온 횟수 / 그 문맥의 전체 횟수
        """
        for n in self._orders_desc():
            if len(recent) >= n:
                counts = self._tables().get(str(n), {}).get(" ".join(recent[-n:]))
                if counts and token in counts:
                    return counts[token] / sum(counts.values())
        return self.FLOOR

    def perplexity(self, sentences):
        """
        문장들의 퍼플렉서티(PPL)를 계산합니다. = exp( 평균( -log p(각 토큰) ) ).
        (문맥이 없는 맨 앞 토큰은 제외하고, 그다음 토큰부터 점수를 매겨요.)
        """
        total_log, total_tokens = 0.0, 0
        for sentence in sentences:
            tokens = self.prepare(self.tokenize(sentence))
            for i in range(1, len(tokens)):
                p = self.token_prob(tokens[:i], tokens[i])
                total_log += math.log(p)
                total_tokens += 1
        if total_tokens == 0:
            return float("inf")
        return math.exp(-total_log / total_tokens)


class Model(NGramLM):
    ORDERS = [1, 2]


DATA_PATH = os.path.join(_VERSION_DIR, "1.data", "data.txt")     # 학습용
VALID_PATH = os.path.join(_VERSION_DIR, "1.data", "valid.txt")   # 검증용 (처음 보는 데이터)
MODEL_PATH = os.path.join(_HERE, "model.json")
