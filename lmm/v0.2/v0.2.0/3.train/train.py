# -*- coding: utf-8 -*-
"""
train.py  (v0.2.0)  -  학습 실행기 (+ 하이퍼파라미터)  · 임베딩 MLP

1.data/data.txt 를 학습해 2.models/model.pt(가중치) + vocab.json(어휘) 을 만듭니다.
v0.1.5 대비 새 하이퍼파라미터: EMBED (임베딩 차원).

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
    m.EMBED = 32               # ★ 임베딩 차원 E (토큰 하나를 몇 차원 벡터로). V≫E 로 압축·공유
    m.HIDDEN = 128             # 은닉층 크기 (2층 MLP)
    m.OPTIMIZER = "adam"       # "sgd" | "momentum" | "adam"
    m.LR = 0.05                # Adam 기준값. SGD/momentum 은 크게(1~10)
    m.EPOCHS = 1_000
    m.BATCH_SIZE = 64
    m.SEED = 1234
    m.WEIGHT_DECAY = 0.0       # 임베딩 병목이 이미 정규화 역할 → 우선 0
    m.INIT = "zeros"           # "zeros"(마지막 층 0=균등 출발) | "default"
    m.LABEL_SMOOTHING = 0.1    # 과신 완화 (임베딩은 과적합이 덜해 0.1 부터)
    # sweep 제안: EMBED ∈ {16, 32, 64}, LABEL_SMOOTHING ∈ {0.05, 0.1, 0.2}
    # ==============================================================

    m.run_train(model.DATA_PATH, model.MODEL_PATH, model.VOCAB_PATH)
