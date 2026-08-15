# -*- coding: utf-8 -*-
"""
train.py  (v0.1.0)  -  학습 실행기 (아주 얇아요!)

실제 학습 코드는 같은 버전 폴더의 2.models/lm.py 에 있어요.
여기서는 그 설정으로 1.data/data.txt 를 학습해 2.models/model.json 을 만들 뿐입니다.

※ 이 버전부터 PyTorch 가 필요합니다:  pip install torch
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "lmmlm_" + os.path.basename(_HERE).replace(".", "_"),
    os.path.join(_HERE, "2.models", "lm.py"),
)
model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model)

if __name__ == "__main__":
    model.Model().run_train(model.DATA_PATH, model.MODEL_PATH)
