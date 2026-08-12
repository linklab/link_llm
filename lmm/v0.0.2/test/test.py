# -*- coding: utf-8 -*-
"""
test.py  -  학습한 모델로 문장을 만들어 보는 실행기 (얇아요!)

v0.0.2 는 앞 '두 단어'를 시작점으로 문장을 만듭니다. (온도 0 = 그리디)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Model, MODEL_PATH


def main():
    lm = Model.load_or_exit(MODEL_PATH)
    print("=== v0.0.2 (앞 단어 2개) 문장 생성 ===")
    for pair in ["나는 아침에", "우리는 함께", "오늘 날씨가", "고양이는 생선을"]:
        print(f"[{pair}] -> {lm.generate(pair)}")


if __name__ == "__main__":
    main()
