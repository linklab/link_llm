# -*- coding: utf-8 -*-
"""
train.py  (v0.1.3)  -  학습 실행기 (+ 하이퍼파라미터)

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
    m.OPTIMIZER = "adam"       # "sgd" | "momentum" | "adam"
    m.LR = 0.05                # Adam 기준값. SGD/momentum 은 크게(1~10)
    m.EPOCHS = 300
    m.BATCH_SIZE = 64
    m.SEED = 1234
    m.WEIGHT_DECAY = 0.0       # 이 작은 모델엔 1e-4 가 너무 강해 underfit(학습 PPL 폭등) → 0 으로
    m.INIT = "zeros"           # "zeros" | "default"  (nn.Linear 초기화)
    m.LABEL_SMOOTHING = 0.1    # ★ 핵심 정규화: 과신을 눌러 '처음 보는 것'에 강하게
                               #   (카운트의 FLOOR 절벽을 피함 = 개수 세기의 add-k 스무딩과 같은 정신)
    # ==============================================================

    m.run_train(model.DATA_PATH, model.MODEL_PATH)
