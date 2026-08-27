# -*- coding: utf-8 -*-
"""
lm.py  (v0.0.9)  -  퍼플렉서티(perplexity) 평가 + 데이터 정비 (NGramLM 상속)

[이 버전이 새로 더한 것]  "얼마나 잘 맞히나?" 를 숫자로 재기
  - token_prob(): 지금 문맥 다음에 특정 토큰이 올 '확률' 을 구함 (백오프 사용)
  - perplexity(): 여러 문장에 대한 퍼플렉서티(PPL) 를 계산
      PPL = exp( 평균( -log p(각 토큰) ) )   →  낮을수록 '잘 맞힘'
      직관: "다음 토큰을 정할 때 평균 몇 개 중에서 헷갈리나". PPL=5 면 5개 중 하나 고르는 수준.

[데이터 정비]  공용 data/ 를 학습용(data.txt)과 검증용(valid.txt)으로 나눴어요.
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

    def next_dist(self, recent):
        """
        지금 문맥(recent) 다음에 올 토큰들의 **확률 분포** {토큰: 확률} 을 돌려줍니다.
        백오프로 **처음 찾은 표**를 그대로 확률로 바꿔요 — generate() 가 다음 토큰을 뽑을 때
        보는 바로 그 표예요. 문맥을 아예 못 찾으면 None.

        ※ token_prob() 은 "그 표에 그 토큰이 없으면 더 짧은 문맥으로 한 번 더" 찾지만,
          여기서는 '모델이 실제로 예측을 뽑는 표' 를 그대로 봅니다(생성과 같은 규칙).
        """
        for n in self._orders_desc():
            if len(recent) >= n:
                counts = self._tables().get(str(n), {}).get(" ".join(recent[-n:]))
                if counts:
                    total = sum(counts.values())
                    return {t: c / total for t, c in counts.items()}
        return None

    def accuracy(self, sentences, top_k=5):
        """
        '다음 토큰 맞히기' 를 **순위**로 재요. PPL 이 "확률을 얼마나 잘 배분했나" 라면
        정확도는 "1등을 얼마나 맞혔나" 예요 — 둘은 순위가 뒤바뀔 수도 있어요.

          top1     : 1등으로 찍은 토큰이 정답인 비율
          topk     : 정답이 상위 k개 안에 든 비율
          coverage : 정답이 후보 목록에 **있기라도 한** 비율
                     (카운트 모델은 표에 없으면 그 토큰을 아예 못 만들어요.
                      = 백오프의 한계가 PPL 의 FLOOR 에 가려지지 않고 그대로 보이는 숫자)

        채점 위치는 perplexity() 와 똑같아요 (문맥이 없는 맨 앞 토큰은 제외).
        """
        n_all = n_top1 = n_topk = n_cov = 0
        for sentence in sentences:
            tokens = self.prepare(self.tokenize(sentence))
            for i in range(1, len(tokens)):
                n_all += 1
                dist = self.next_dist(tokens[:i])
                if not dist or tokens[i] not in dist:
                    continue                      # 후보에 없음 = 맞힐 방법이 없음
                n_cov += 1
                ranked = [t for t, _ in sorted(dist.items(), key=lambda kv: -kv[1])]
                if ranked[0] == tokens[i]:
                    n_top1 += 1
                if tokens[i] in ranked[:top_k]:
                    n_topk += 1
        if n_all == 0:
            return {"top1": 0.0, "topk": 0.0, "coverage": 0.0, "n": 0, "k": top_k}
        return {"top1": n_top1 / n_all, "topk": n_topk / n_all,
                "coverage": n_cov / n_all, "n": n_all, "k": top_k}

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


DATA_PATH = os.path.join(_DATA_DIR, "data.txt")     # 학습용
VALID_PATH = os.path.join(_DATA_DIR, "valid.txt")   # 검증용 (처음 보는 데이터)
MODEL_PATH = os.path.join(_HERE, "model.json")
