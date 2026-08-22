# -*- coding: utf-8 -*-
"""
train.py  (v0.1.1)  -  학습 실행기 (+ 하이퍼파라미터)

하이퍼파라미터를 여기서 정해 모델에 넘기고, 1.data/data.txt 를 학습해
2.models/model.json 을 만듭니다.

※ PyTorch 필요:  pip install torch
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
    m.HIDDEN = 128       # 은닉층 크기 (2층 MLP)
    m.LR = 1.0           # 학습률 (2층 MLP 는 예전 선형(10)보다 작게)
    m.EPOCHS = 1_000       # 전체 데이터를 몇 번 반복(에폭)
    m.BATCH_SIZE = 64    # 한 배치에 담는 (앞,다음) 짝의 수
    m.SEED = 1234        # 초기 가중치 + DataLoader 셔플을 재현 가능하게
    # ==============================================================

    m.run_train(model.DATA_PATH, model.MODEL_PATH, model.VOCAB_PATH)
