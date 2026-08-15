# -*- coding: utf-8 -*-
"""
train.py  -  학습 실행기 (아주 얇아요!)

실제 학습 코드는 같은 버전 폴더의 base.py (와 그것이 물려받는 이전 버전들)에 있고,
이 버전이 '무엇이 다른지' 는 ../model.py 의 Model 클래스에 있어요.
여기서는 그 설정으로 1.data/data.txt 를 학습해 models/model.json 을 만들 뿐입니다.
"""
import os
import importlib.util

# 같은 버전 폴더(v0.0.X)의 models/lm.py 를 '파일 경로'로 불러옵니다.
# (폴더 이름에 '.' 이 있어 일반 import 가 안 되므로 importlib 로 로딩해요.)
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "lmmlm_" + os.path.basename(_HERE).replace(".", "_"),
    os.path.join(_HERE, "2.models", "lm.py"),
)
model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model)

if __name__ == "__main__":
    model.Model().run_train(model.DATA_PATH, model.MODEL_PATH)
