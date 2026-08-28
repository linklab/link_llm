# -*- coding: utf-8 -*-
"""
test.py  (v0.0.8)  -  대화 형식 + 멀티턴 문맥 데모

lm.chat(입력, 기록) 을 부르면 '봇의 답' 만 돌려줘요.
기록(history)을 쌓아가며 여러 번 주고받는 '멀티턴' 대화를 흉내 냅니다.
"""
import os
import importlib.util

# 같은 버전 폴더의 model.py 를 '파일 경로'로 불러옵니다. (dot 폴더라 일반 import 불가)
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
    print("=== v0.0.8 (긴 문맥으로 문장 이어가기) ===\n")

    # 산문으로 '사전학습'한 모델이라, 씨앗을 주면 그다음을 이어 씁니다.
    # (역할 토큰/대화 형식 자체는 v0.5 SFT 단계에서 다뤄요.)
    for seed in ["아침 일찍", "봄 바람", "나는 조용한"]:
        print(f"  [씨앗] {seed}")
        print(f"  [생성] {lm.generate(seed, temperature=0.3)}\n")


if __name__ == "__main__":
    main()
