# -*- coding: utf-8 -*-
"""
train.py  -  학습 실행기 (아주 얇아요!)

실제 학습 코드는 같은 버전 폴더의 base.py (와 그것이 물려받는 이전 버전들)에 있고,
이 버전이 '무엇이 다른지' 는 ../model.py 의 Model 클래스에 있어요.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Model, DATA_PATH, MODEL_PATH

if __name__ == "__main__":
    Model().run_train(DATA_PATH, MODEL_PATH)
