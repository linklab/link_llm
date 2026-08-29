# -*- coding: utf-8 -*-
"""
train.py  (v0.1.2)  -  학습 실행기 (+ 하이퍼파라미터)

하이퍼파라미터를 여기서 정해 모델에 넘기고, 공용 data/data.txt 를 학습해
0.model/model.json 을 만듭니다.

※ PyTorch 필요:  pip install torch
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "llmlm_" + os.path.basename(_HERE).replace(".", "_"),
    os.path.join(_HERE, "0.model", "lm.py"),
)
model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model)

if __name__ == "__main__":
    m = model.Model()

    # ================= 하이퍼파라미터 (여기서 조정) =================
    # 아래 값은 5개 시드 · 에폭별 검증 PPL 을 재서 고른 설정이에요 (조기 종료 시 검증 PPL 39.5).
    m.OPTIMIZER = "adam"   # "sgd" | "momentum" | "adam"  (골라 쓰기)
    m.HIDDEN = 256         # 은닉층 크기 (2층 MLP)
    m.LR = 0.0003          # ★ Adam 은 1e-4~1e-3 대. 0.05 는 50배 이상 커서 발산해요
    m.EPOCHS = 1_500       # ★ 2,180개 짝뿐이라 금방 과적합. 100~160 이 평평한 바닥
    m.BATCH_SIZE = 64
    m.SEED = 1234
    m.DEVICE = "auto"    # "auto"=애플 실리콘 GPU(MPS) 있으면 사용, 없으면 CPU. "cpu"/"mps" 로 못박기 가능
    # --- 조기 종료 (early stopping) ---
    # 학습 손실(Loss)이 PATIENCE 에폭 동안 나아지지 않으면 멈추고, **가장 좋았던 가중치**로 되돌려요.
    # 그래서 EPOCHS 는 이제 '정확히 맞춰야 하는 값'이 아니라 넉넉한 **상한**이면 됩니다.
    m.EARLY_STOPPING = True    # False 면 EPOCHS 를 끝까지 돕니다
    m.PATIENCE = 20            # 조기 종료 인내 에폭(통일)
    m.MIN_DELTA = 0.0          # 개선으로 인정할 최소 폭(통일)
    # ==============================================================

    m.run_train(model.DATA_PATH, model.MODEL_PATH, model.VOCAB_PATH, model.VALID_PATH)
