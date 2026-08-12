# -*- coding: utf-8 -*-
"""
test.py  -  학습한 모델로 문장을 만들어 보는 실행기 (얇아요!)

v0.0.4 의 핵심은 '백오프' 예요. 아래처럼 1단어만 입력해도 시작할 수 있어요.
(온도 0 = 그리디: 가장 많이 나온 단어를 고름)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Model, MODEL_PATH


def main():
    lm = Model.load_or_exit(MODEL_PATH)

    print("=== v0.0.4 (백오프) ===\n")
    print("[1단어만 입력해도 시작돼요 — v0.0.3 에서는 안 되던 것!]")
    for word in ["나는", "우리는", "오늘", "고양이는"]:
        print(f"   [{word}] -> {lm.generate(word)}")

    print("\n[2단어로 시작]")
    for pair in ["나는 아침에", "우리는 함께", "겨울에는 눈이"]:
        print(f"   [{pair}] -> {lm.generate(pair)}")


if __name__ == "__main__":
    main()
