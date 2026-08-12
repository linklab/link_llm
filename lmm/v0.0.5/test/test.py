# -*- coding: utf-8 -*-
"""
test.py  -  학습한 모델로 문장을 만들어 보는 실행기 (얇아요!)

v0.0.5 의 핵심은 '온도(temperature)' 예요. 같은 시작이라도 온도에 따라 답이 달라져요.
  - 낮은 온도(0.3): 뻔하고 안정적   / 높은 온도(1.5): 다양하고 뜻밖
"""
import os
import importlib.util

# 같은 버전 폴더의 model.py 를 '파일 경로'로 불러옵니다. (dot 폴더라 일반 import 불가)
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "lmmmodel_" + os.path.basename(_HERE).replace(".", "_"),
    os.path.join(_HERE, "model.py"),
)
_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model)
Model = _model.Model
MODEL_PATH = _model.MODEL_PATH


def main():
    lm = Model.load_or_exit(MODEL_PATH)
    start = "우리는 함께"
    print("=== v0.0.5 (확률 + 온도) ===")
    print(f"시작: '{start}' — 온도에 따라 다른 문장이 나와요.\n")
    for temp in [0.3, 1.0, 1.5]:
        print(f"[온도 {temp}]")
        for _ in range(3):
            print("   ", lm.generate(start, temperature=temp))
        print()


if __name__ == "__main__":
    main()
