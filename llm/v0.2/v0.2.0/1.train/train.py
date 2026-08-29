# -*- coding: utf-8 -*-
"""
train.py  (v0.2.0)  -  학습 실행기 (+ 하이퍼파라미터)  · 임베딩 MLP

공용 data/data.txt 를 학습해 0.model/model.pt(가중치) + vocab.json(어휘) 을 만듭니다.
v0.1.5 대비 새 하이퍼파라미터: EMBED (임베딩 차원).

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
    # 아래 값은 8개 시드 · 에폭별 검증 PPL 을 재서 고른 설정이에요.
    #   조기 종료 시 검증 PPL 32.76 (카운트 기준선 v0.0.9 = 34.39 아래 ✅)
    m.EMBED = 256              # ★ 임베딩 차원 E. 데이터가 282문장뿐이라 E 를 키울수록 좋아요
                               #   (E=32→37.6, 64→36.9, 128→35.3, 256→32.9, 512→34.5)
    m.HIDDEN = 256             # 은닉층 크기 (2층 MLP)
    m.OPTIMIZER = "adam"       # "sgd" | "momentum" | "adam"
    m.LR = 0.0003              # ★ Adam 은 1e-4~1e-3 대. 0.05 는 50배 이상 커서 발산해요
    m.EPOCHS = 1_500              # ★ 2,180개 짝뿐이라 금방 과적합. 40~60 이 평평한 바닥
    m.BATCH_SIZE = 64
    m.SEED = 1234
    m.DEVICE = "auto"    # "auto"=애플 실리콘 GPU(MPS) 있으면 사용, 없으면 CPU. "cpu"/"mps" 로 못박기 가능
    m.WEIGHT_DECAY = 0.0       # 실측: 1e-4 부터 이미 검증 PPL 이 나빠져요 → 0
    m.INIT = "zeros"           # "zeros"(마지막 층 0=균등 출발) | "default"
    m.LABEL_SMOOTHING = 0.0    # 실측: FLOOR(1e-4) 가 이미 바닥을 깔아줘 스무딩은 손해
    # sweep 제안: EMBED ∈ {128, 256, 384} × LR ∈ {2e-4, 3e-4, 5e-4}
    #             (EPOCHS 는 조기 종료가 알아서 정하므로 더 이상 손댈 필요 없어요)
    # --- 조기 종료 (early stopping) ---
    # 학습 손실(Loss)이 PATIENCE 에폭 동안 나아지지 않으면 멈추고, **가장 좋았던 가중치**로 되돌려요.
    # 그래서 EPOCHS 는 이제 '정확히 맞춰야 하는 값'이 아니라 넉넉한 **상한**이면 됩니다.
    m.EARLY_STOPPING = True    # False 면 EPOCHS 를 끝까지 돕니다
    m.PATIENCE = 20            # 조기 종료 인내 에폭(통일)
    m.MIN_DELTA = 0.0          # 개선으로 인정할 최소 폭(통일)
    # ==============================================================

    m.run_train(model.DATA_PATH, model.MODEL_PATH, model.VOCAB_PATH, model.VALID_PATH)
