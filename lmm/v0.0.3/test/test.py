# -*- coding: utf-8 -*-
"""
test.py  -  학습한 모델로 문장을 만들어 보는 실행기 (얇아요!)

v0.0.3 은 <END> 를 만나면 문장을 자연스럽게 끝냅니다. (온도 0 = 그리디)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Model, MODEL_PATH


def main():
    lm = Model.load_or_exit(MODEL_PATH)
    print("=== v0.0.3 (문장 끝 <END> 학습) 문장 생성 ===")
    for pair in ["나는 아침에", "우리는 함께", "오늘 날씨가", "겨울에는 눈이"]:
        print(f"[{pair}] -> {lm.generate(pair)}")


if __name__ == "__main__":
    main()
