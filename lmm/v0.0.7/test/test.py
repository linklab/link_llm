# -*- coding: utf-8 -*-
"""
test.py  (v0.0.7)  -  top-k / top-p 샘플링 효과 비교

같은 문맥 '나는 오늘' 뒤에는 후보가 15곳이나 있고, 분포가 한쪽으로 쏠려 있어요:
  학교에(5), 공원에(3), 도서관에(2), 그리고 병원에/시장에/바다에/... (각 1)  ← 긴 꼬리

- 무제한(top_k=0, top_p=1.0): 꼬리의 드문 장소(동물원에, 수영장에 ...)도 종종 튀어나와요.
- top_k=3            : 확률 상위 3곳(학교에·공원에·도서관에)만 후보로 -> 안정적.
- top_p=0.6 (nucleus): 누적확률 60%까지의 후보만 -> 역시 꼬리를 잘라냄.

같은 온도(1.0)라도 '자르기' 설정에 따라 결과의 다양성이 확 달라지는 걸 볼 수 있어요.
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


def show(lm, label, temperature=1.0, **kwargs):
    print(f"[{label}]")
    for _ in range(8):
        print("   ", lm.generate("나는 오늘", temperature=temperature, **kwargs))
    print()


def main():
    lm = Model.load_or_exit(MODEL_PATH)
    print("=== v0.0.7 (top-k / top-p 샘플링) ===")
    print("문맥 '나는 오늘' — 온도는 1.0 으로 고정, 자르기 설정만 바꿔서 비교\n")

    show(lm, "무제한 (top_k=0, top_p=1.0)")                 # 꼬리까지 다 나옴 -> 다양(때때로 엉뚱)
    show(lm, "top_k=3", top_k=3)                            # 상위 3곳만
    show(lm, "top_p=0.6 (nucleus)", top_p=0.6)             # 누적확률 60%까지


if __name__ == "__main__":
    main()
