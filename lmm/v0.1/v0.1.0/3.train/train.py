# -*- coding: utf-8 -*-
"""
train.py  (v0.1.0)  -  학습 실행기 (+ 하이퍼파라미터)

하이퍼파라미터를 여기서 정해 모델에 넘기고, 1.data/data.txt 를 학습해
2.models/model.json 을 만듭니다.

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
    m = model.Model()

    # ================= 하이퍼파라미터 (여기서 조정) =================
    m.LR = 10.0          # 학습률 (full-batch 경사하강이라 크게 잡아요)
    m.EPOCHS = 10_000    # 전체 데이터를 몇 번 반복할지
    m.SEED = 1234        # 초기 W 를 재현 가능하게
    # ==============================================================

    m.run_train(model.DATA_PATH, model.MODEL_PATH)
