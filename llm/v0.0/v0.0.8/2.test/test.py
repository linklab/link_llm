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
    print("=== v0.0.8 (대화 형식 + 멀티턴) ===\n")

    conversation = ["안녕", "이름이 뭐야?", "노래 불러줘", "고마워", "잘 자"]
    history = []
    for user_msg in conversation:
        reply = lm.chat(user_msg, history, temperature=0.3)
        print(f"  나 : {user_msg}")
        print(f"  봇 : {reply}\n")
        history.append((user_msg, reply))   # 기록에 쌓아 다음 턴 문맥으로 사용


if __name__ == "__main__":
    main()
