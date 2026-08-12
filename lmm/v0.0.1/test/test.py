# -*- coding: utf-8 -*-
"""
test.py  -  학습한 모델로 문장을 만들어 보는 실행기 (얇아요!)

문장 만들기 코드는 같은 폴더의 base.py (NGramLM.generate) 에 있어요.
v0.0.1 은 '가장 많이 나온 단어'를 고릅니다. (온도 0 = 그리디)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Model, MODEL_PATH


def main():
    lm = Model.load_or_exit(MODEL_PATH)
    print("=== v0.0.1 (앞 단어 1개) 문장 생성 ===")
    for word in ["나는", "너는", "우리는", "오늘", "고양이는"]:
        print(f"[{word}] -> {lm.generate(word)}")


if __name__ == "__main__":
    main()
