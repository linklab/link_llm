# -*- coding: utf-8 -*-
"""
test.py  -  학습한 모델로 문장을 만들어 보는 실행기 (얇아요!)

v0.0.6 은 문장부호를 따로 떼어 학습했기 때문에, 문장부호가 자연스럽게 나옵니다.
  예) "오늘 날씨가 좋습니다!", "너는 아침에 무엇을 먹니?"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Model, MODEL_PATH


def main():
    lm = Model.load_or_exit(MODEL_PATH)
    print("=== v0.0.6 (백오프 + 온도 + 문장부호 토크나이저) ===\n")
    for start in ["오늘 날씨가", "나는 코딩을", "우리는 함께", "너는 아침에"]:
        print(f"[{start}]")
        for _ in range(3):
            print("   ", lm.generate(start, temperature=0.7))
        print()


if __name__ == "__main__":
    main()
