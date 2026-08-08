# -*- coding: utf-8 -*-
"""
test.py  -  학습한 모델로 "문장을 만들어(생성)" 보는 프로그램

[핵심 아이디어]
train.py 에서 만든 표(model.json)를 불러옵니다.
그 표에는 "어떤 단어 뒤에 어떤 단어가 몇 번 나왔는지" 가 적혀 있어요.

문장 만들기 방법:
  1. 시작 단어를 하나 정한다. 예) "나는"
  2. 표를 보고 "나는" 뒤에 가장 자주 나온 단어를 고른다.
  3. 그 단어를 이어 붙이고, 다시 그 단어의 다음 단어를 고른다.
  4. 이걸 여러 번 반복하면 문장이 만들어져요!

가장 자주 나온 단어를 고르기 때문에, 배운 문장과 비슷한 문장이 나옵니다.
"""

import json
import os

# ----------------------------------------------------------------------
# 경로 정하기: 저장된 모델 파일을 찾습니다.
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))          # test 폴더 위치
PROJECT = os.path.dirname(HERE)                           # link_lmm 폴더 위치
MODEL_PATH = os.path.join(PROJECT, "models", "model.json")


def load_model(path):
    """저장해둔 모델(표)을 불러옵니다."""
    if not os.path.exists(path):
        print("모델 파일이 없어요! 먼저 train/train.py 를 실행해 주세요.")
        raise SystemExit(1)   # 프로그램을 여기서 멈춤

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_next_word(model, word):
    """
    지금 단어 뒤에 나올 '다음 단어'를 고릅니다.
    규칙: 가장 많이 나왔던(횟수가 가장 큰) 단어를 고른다.
    다음 단어가 없으면 None(없음)을 돌려줍니다.
    """
    if word not in model:
        return None

    next_words = model[word]      # 예) {"밥을": 2, "학교에": 1}
    if not next_words:
        return None

    # 횟수(value)가 가장 큰 단어를 찾습니다.
    best_word = None
    best_count = -1
    for candidate, count in next_words.items():
        if count > best_count:
            best_count = count
            best_word = candidate
    return best_word


def generate(model, start_word, max_length=8):
    """
    start_word 로 시작해서 최대 max_length 개의 단어로 문장을 만듭니다.
    """
    result = [start_word]         # 문장의 첫 단어
    current = start_word

    for _ in range(max_length - 1):
        nxt = pick_next_word(model, current)
        if nxt is None:           # 더 이을 단어가 없으면 멈춤
            break
        result.append(nxt)
        current = nxt

    return " ".join(result)       # 단어들을 공백으로 이어 붙여 문장으로


def main():
    model = load_model(MODEL_PATH)

    # 이 단어들로 각각 문장을 만들어 봅니다. 마음대로 바꿔보세요!
    start_words = ["나는", "너는", "우리는", "오늘", "고양이는"]

    print("=== 나만의 작은 언어 모델 v0.0.1 문장 생성 결과 ===")
    for w in start_words:
        sentence = generate(model, w)
        print(f"[{w}] -> {sentence}")


if __name__ == "__main__":
    main()
