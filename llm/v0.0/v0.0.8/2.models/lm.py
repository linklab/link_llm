# -*- coding: utf-8 -*-
"""
lm.py  (v0.0.8)  -  대화 형식 + 멀티턴 문맥 (NGramLM 상속 + 설정)

[이 버전이 새로 더한 것]  ChatGPT 같은 '대화 골격'
  - 역할 토큰: <사용자> / <봇> — 누가 말한 차례인지 표시해요.
  - build_chat_context(): 대화 기록 + 이번 입력을 '<사용자> ... <봇> ...' 형식의 토큰으로.
  - chat(): 그 문맥을 이어 붙여 '봇의 답' 만 만들어 돌려줍니다. (멀티턴)

[학습 데이터도 대화 형식]  공용 data/data.txt 의 각 줄이
  "<사용자> 안녕 <봇> 안녕하세요 !"  처럼 한 번의 대화(주고받기)로 되어 있어요.
그래서 모델은 "<봇> 다음엔 답이 오고 <END> 로 끝난다" 를 배웁니다.

[멀티턴이란?]  이전 대화(기록)를 문맥으로 이어받아 다음 답을 만드는 것.
  <사용자> 안녕 <봇> 안녕하세요 ! <사용자> 이름이 뭐야 ? <봇> ___
  처럼 지금까지의 대화를 통째로 문맥으로 삼아요.

(참고: 우리 모델은 앞 1~2토큰만 보는 n-gram이라 '먼 과거'는 잘 못 봐요.
 질문이 모두 '?' 로 끝나면 문맥 '? <봇>' 이 겹쳐 답이 섞일 수 있어요.
 진짜 긴 대화 기억은 나중에 트랜스포머에서 다뤄집니다.)
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


class NGramLM(_load_prev("v0.0.7")):
    USER = "<사용자>"      # 사용자 차례를 나타내는 역할 토큰
    BOT = "<봇>"          # 봇(모델) 차례를 나타내는 역할 토큰

    def build_chat_context(self, history, user_message):
        """
        대화 기록(history) + 이번 입력을 '<사용자> ... <봇> ...' 형식의 토큰 리스트로 만듭니다.
        history 는 [(사용자말, 봇답), ...] 형태예요.
        """
        tokens = []
        for pair in (history or []):
            if len(pair) >= 2:
                tokens += [self.USER] + self.tokenize(pair[0]) + [self.BOT] + self.tokenize(pair[1])
        tokens += [self.USER] + self.tokenize(user_message) + [self.BOT]  # 이번 입력 + '봇 차례'
        return tokens

    def chat(self, user_message, history=None, temperature=0.0, top_k=0, top_p=1.0):
        """
        대화 기록과 이번 입력을 문맥으로, '봇의 답' 만 만들어 돌려줍니다.
        (문장 만들기 자체는 물려받은 next_token/choose 를 그대로 재사용해요.)
        """
        self._top_k, self._top_p = top_k, top_p     # v0.0.7 choose 가 참조
        recent = self.build_chat_context(history, user_message)

        reply = []
        for _ in range(self.MAX_LENGTH):
            nxt = self.next_token(recent, temperature)
            # 답이 끝났거나(<END>), 다음 차례(<사용자>/<봇>)가 시작되면 멈춤
            if nxt is None or self.is_end(nxt) or nxt in (self.USER, self.BOT):
                break
            recent.append(nxt)
            reply.append(nxt)

        return self.detokenize(reply)


class Model(NGramLM):
    ORDERS = [1, 2]


DATA_PATH = os.path.join(_DATA_DIR, "data.txt")
MODEL_PATH = os.path.join(_HERE, "model.json")
