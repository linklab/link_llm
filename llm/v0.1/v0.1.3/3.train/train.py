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
    "llmlm_" + os.path.basename(_HERE).replace(".", "_"),
    os.path.join(_HERE, "2.models", "lm.py"),
)
model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model)

if __name__ == "__main__":
    m = model.Model()

    # ================= 하이퍼파라미터 (여기서 조정) =================
    # 아래 값은 5개 시드 · 에폭별 검증 PPL 을 재서 고른 설정이에요 (검증 PPL 39.5 ± 0.4).
    m.OPTIMIZER = "adam"       # "sgd" | "momentum" | "adam"
    m.HIDDEN = 256             # 은닉층 크기 (2층 MLP)
    m.LR = 0.0003              # ★ Adam 은 1e-4~1e-3 대. 0.05 는 50배 이상 커서 발산해요
    m.EPOCHS = 130             # ★ 2,180개 짝뿐이라 금방 과적합. 100~160 이 평평한 바닥
    m.BATCH_SIZE = 64
    m.SEED = 1234
    m.WEIGHT_DECAY = 0.0       # 실측: 1e-4 부터 이미 검증 PPL 이 나빠져요 → 0
    m.INIT = "zeros"           # "zeros" | "default"  (nn.Linear 초기화)
    m.LABEL_SMOOTHING = 0.0    # ★ 실측: 0.1 을 켜면 검증 PPL 이 39.5 → 41.1 로 나빠져요.
                               #   이 평가에서는 perplexity() 가 FLOOR(1e-4)로 이미 바닥을
                               #   깔아줘서(=add-k 스무딩과 같은 역할) 스무딩이 중복이거든요.
                               #   개념을 눈으로 보려면 0.1/0.2 로 올려 PPL 변화를 확인해 보세요.
    # ==============================================================

    m.run_train(model.DATA_PATH, model.MODEL_PATH, model.VOCAB_PATH)
