# -*- coding: utf-8 -*-
"""
test.py  -  학습한 모델로 문장을 만들어 보는 실행기 (얇아요!)

v0.0.3 은 <END> 를 만나면 문장을 자연스럽게 끝냅니다. (온도 0 = 그리디)
"""
import os
import importlib.util

# 같은 버전 폴더의 models/lm.py 를 '파일 경로'로 불러옵니다. (dot 폴더라 일반 import 불가)
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "llmlm_" + os.path.basename(_HERE).replace(".", "_"),
    os.path.join(_HERE, "0.model", "lm.py"),
)
_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model)
Model = _model.Model
MODEL_PATH = _model.MODEL_PATH


def main():
    lm = Model.load_or_exit(MODEL_PATH)
    print("=== v0.0.3 (문장 끝 <END> 학습) 문장 생성 ===")
    for pair in ["나는 아침에", "우리는 함께", "오늘 날씨가", "겨울에는 눈이"]:
        print(f"[{pair}] -> {lm.generate(pair)}")


if __name__ == "__main__":
    main()
