# -*- coding: utf-8 -*-
"""
lm.py  (v0.0.6)  -  토크나이저: 문장부호 + 조사 분리 (NGramLM 상속 + 설정)

[이 버전이 새로 더한 것]
  텍스트를 **어떤 단위로 자를지**를 정하는 버전이에요. v0.0.1 의 `whitespace`(그냥 띄어쓰기)를
  두 단계로 다듬습니다.

    ① 문장부호 분리 : "맛있었다." → "맛있었다" + "."
                      안 그러면 "맛있었다." 와 "맛있었다" 가 서로 다른 토큰이 돼요.
    ② 조사 분리     : "학교를"   → "학교" + "##를"
                      한국어는 교착어라 명사 뒤에 조사가 달라붙어요. 안 나누면
                      "학교를 / 학교는 / 학교에" 가 **서로 무관한 세 개의 토큰**이 됩니다.

[왜 ② 가 필요한가 — 실측]
  조사를 안 나누면 어휘의 21.8% 가 같은 말의 변이형으로 채워져요.
      [바다] 바다로(42) · 바다에(41) · 바다를(34) · 바다에서(27) · 바다는(1)
  모델은 이 다섯을 **아무 관계 없는 다섯 단어**로 배웁니다. 실제로 v0.2.5 임베딩에서
  "영화관에" 의 최근접이웃은 동물원에·미술관에(같은 조사)였고, 정작 형제인 "영화관을" 은
  유사도 -0.01 로 남남이었어요. 조사가 어간보다 강한 신호가 돼버린 겁니다.

[##  표시]
  조사 토큰에는 "##" 를 붙여 **앞 토큰에 붙는 조각**임을 표시해요 (BERT/WordPiece 방식).
  덕분에 detokenize 가 "학교" + "##를" 을 공백 없이 "학교를" 로 정확히 되돌립니다.

[규칙 기반의 한계 — 숨기지 않아요]
  사전 없이 접미 규칙으로만 자르므로 **오분할이 남습니다**.
      뛰어노는 → 뛰어노 + ##는   (동사 어간은 '뛰어놀-' 이라 틀림)
  규칙으로는 예외를 다 못 막아요. 이게 v0.4.0 에서 **데이터로 배우는 BPE** 로 넘어가는 이유예요.

백오프·온도·문장끝 등은 이전 버전에서 물려받아요.
"""

import os
import re
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERSION_DIR = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_VERSION_DIR)))  # 저장소 루트
_DATA_DIR = os.path.join(_ROOT, "data")                                  # 모든 버전 공용 데이터


def _load_prev(prev_version):
    group = prev_version.rsplit(".", 1)[0]                    # "v0.0.9" -> "v0.0" (마이너 그룹)
    llm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../llm 루트
    path = os.path.join(llm_dir, group, prev_version, "0.model", "lm.py")
    spec = importlib.util.spec_from_file_location("llmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NGramLM


class NGramLM(_load_prev("v0.0.5")):
    PUNCT = {".", ",", "!", "?"}

    # 앞 토큰에 붙는 조각임을 나타내는 표시 (BERT/WordPiece 의 "##" 와 같은 뜻)
    SUFFIX_MARK = "##"

    # 떼어낼 조사 목록. **긴 것부터** 봐야 "에서" 를 "에"+"서" 로 잘못 자르지 않아요.
    JOSA = ("에서", "으로", "에게", "까지", "부터", "보다", "처럼",
            "를", "을", "가", "이", "는", "은", "에", "로", "와", "과", "도", "만", "의")

    # 조사를 뗀 뒤 남는 어간의 최소 길이. 1글자만 남으면 대개 오분할이에요
    # (예: "포도는" 의 "포도" 는 "도" 로 끝나지만 남는 게 "포" 뿐이라 자르지 않아요).
    MIN_STEM = 2

    # 규칙만으로는 못 거르는 예외 — 조사처럼 끝나지만 조사가 아닌 말들.
    #   · 동사·형용사 관형형(-는/-은): 어간이 동사라 자르면 "뛰어노" 같은 쓰레기 조각이 생겨요
    #   · 부사: "느지막이" 의 "이" 는 주격조사가 아니에요
    # 이런 목록이 필요하다는 것 자체가 규칙 기반의 한계예요 → v0.4.0 에서 BPE 로 갑니다.
    NO_SPLIT = frozenset({
        # 동사·형용사 관형형
        "뛰어노는", "돌아오는", "좋아하는", "어울리는", "키우는", "만드는", "기르는", "배우는",
        "경기하는", "부르는", "탐험하는", "지키는", "치료하는", "달리는", "연구하는", "넓히는",
        "알리는", "튕기는", "기다리는", "연주하는", "지내는", "가르는", "정돈되는", "다루는",
        "표현하는", "겨루는", "빠지는", "그리는", "북적이는", "돌보는", "따르는", "차오르는",
        "뒤뚱거리는", "물리는", "신나는", "맛있는",
        # 부사
        "느지막이", "틈틈이", "촉촉이", "저절로",
        # 조사처럼 끝나지만 그게 단어의 일부인 명사
        # (안 막으면 "거북이가" → 거북+##이+##가 처럼 어간까지 쪼개져요)
        "거북이", "고양이", "부엉이", "원숭이", "호랑이", "떡볶이", "오토바이", "물놀이",
        "태권도", "수정과", "작곡가", "사진작가", "오랜만",
    })

    def split_josa(self, word):
        """
        단어 하나를 [어간, ##조사] 로 나눠요. 나눌 수 없으면 [단어] 그대로.

        두 가지를 조심합니다.
          · **긴 조사 우선, 그리고 실패하면 멈춤** — "곁으로" 는 "으로" 가 걸리지만 남는 게
            "곁"(1글자)뿐이라 포기해요. 여기서 더 짧은 "로" 로 내려가면 "곁으"+"로" 라는
            엉뚱한 조각이 생깁니다.
          · **겹조사는 재귀로** — "오후에는" → "오후" + ##에 + ##는
        """
        if word in self.NO_SPLIT:
            return [word]
        for josa in self.JOSA:                       # JOSA 는 긴 것부터 정렬돼 있어요
            if not word.endswith(josa):
                continue
            if len(word) - len(josa) < self.MIN_STEM:
                break                                # 남는 어간이 너무 짧으면 그냥 포기
            return self.split_josa(word[:-len(josa)]) + [self.SUFFIX_MARK + josa]
        return [word]

    def tokenize(self, text):
        """① 문장부호를 떼고 ② 각 단어에서 조사를 떼어냅니다."""
        words = re.findall(r"[.,!?]|[^\s.,!?]+", text)          # ① 문장부호 분리
        tokens = []
        for w in words:
            tokens.extend([w] if w in self.PUNCT else self.split_josa(w))   # ② 조사 분리
        return tokens

    def detokenize(self, tokens):
        """토큰을 문장으로 되돌려요. 문장부호와 ##조각은 앞 토큰에 딱 붙입니다."""
        text = ""
        for t in tokens:
            if t.startswith(self.SUFFIX_MARK):       # ##를 -> 앞 단어에 공백 없이
                text += t[len(self.SUFFIX_MARK):]
            elif t in self.PUNCT:
                text += t
            else:
                text += (" " if text else "") + t
        return text

    def tokenizer_name(self):
        return "punct+josa"


class Model(NGramLM):
    ORDERS = [1, 2]


DATA_PATH = os.path.join(_DATA_DIR, "pretrain", "train.txt")
MODEL_PATH = os.path.join(_HERE, "model.json")
