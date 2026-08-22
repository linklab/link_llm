# -*- coding: utf-8 -*-
"""
train.py  (v0.1.4)  -  학습 실행기 (+ 하이퍼파라미터)  · 2토큰 문맥 신경망

하이퍼파라미터를 여기서 정해 모델에 넘기고, 1.data/data.txt 를 학습해
2.models/model.json (2토큰 문맥 가중치 W2) 을 만듭니다.

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
    m.HIDDEN = 128             # 은닉층 크기 (2층 MLP)
    m.LR = 0.05                # Adam 기준값. SGD/momentum 은 크게(1~10)
    m.EPOCHS = 1_000
    m.BATCH_SIZE = 64
    m.SEED = 1234
    m.WEIGHT_DECAY = 0.0       # 이 모델들엔 1e-4 에도 underfit 할 만큼 민감 → 0 (아래 sweep 참고)
    m.INIT = "zeros"           # "zeros" | "default"  (nn.Linear 초기화)
    m.LABEL_SMOOTHING = 0.2    # ★ 2토큰은 용량이 2배라 과적합↑ → 1토큰(0.1)보다 세게(0.2)
                               #   과신을 더 눌러 '처음 보는 2토큰 조합'에 강하게
    # sweep 제안: LABEL_SMOOTHING ∈ {0.15, 0.2, 0.3}, (원하면) WEIGHT_DECAY ∈ {0, 1e-5, 5e-5}
    # ==============================================================

    m.run_train(model.DATA_PATH, model.MODEL_PATH, model.VOCAB_PATH)
