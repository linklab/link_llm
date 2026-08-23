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
    "llmlm_" + os.path.basename(_HERE).replace(".", "_"),
    os.path.join(_HERE, "2.models", "lm.py"),
)
model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model)

if __name__ == "__main__":
    m = model.Model()

    # ================= 하이퍼파라미터 (여기서 조정) =================
    # 아래 값은 8개 시드 · 에폭별 검증 PPL 을 재서 고른 설정이에요.
    #   검증 PPL 33.1 ± 1.2  (카운트 기준선 v0.0.9 = 34.39 아래 ✅)
    m.EMBED = 256              # ★ 임베딩 차원 E. 데이터가 282문장뿐이라 E 를 키울수록 좋아요
                               #   (E=32→37.6, 64→36.9, 128→35.3, 256→32.9, 512→34.5)
    m.HIDDEN = 256             # 은닉층 크기 (2층 MLP)
    m.OPTIMIZER = "adam"       # "sgd" | "momentum" | "adam"
    m.LR = 0.0003              # ★ Adam 은 1e-4~1e-3 대. 0.05 는 50배 이상 커서 발산해요
    m.EPOCHS = 50              # ★ 2,180개 짝뿐이라 금방 과적합. 40~60 이 평평한 바닥
    m.BATCH_SIZE = 64
    m.SEED = 1234
    m.WEIGHT_DECAY = 0.0       # 실측: 1e-4 부터 이미 검증 PPL 이 나빠져요 → 0
    m.INIT = "zeros"           # "zeros"(마지막 층 0=균등 출발) | "default"
    m.LABEL_SMOOTHING = 0.0    # 실측: FLOOR(1e-4) 가 이미 바닥을 깔아줘 스무딩은 손해
    # sweep 제안: LR ∈ {2e-4, 3e-4, 5e-4} × EPOCHS ∈ {30, 50, 70} (LR↑이면 EPOCHS↓)
    # ==============================================================

    m.run_train(model.DATA_PATH, model.MODEL_PATH, model.VOCAB_PATH)
