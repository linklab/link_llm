# -*- coding: utf-8 -*-
"""
train.py  (v0.2.3)  -  학습 실행기 (+ 하이퍼파라미터)  · 일반화·튜닝

공용 data/pretrain/train.txt 를 학습해 0.model/model.pt + vocab.json 을 만듭니다.
v0.2.2 대비 더한 것: dropout · weight tying · LR 스케줄.

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
    m.BLOCK_SIZE = 3           # 문맥 길이 N (v0.2.1~2.2 와 같은 조건)
    m.EMBED = 128              # ★ 임베딩 차원 E — weight tying 을 쓰려면 HIDDEN 과 같아야 해요
    m.HIDDEN = 128             # ★ 은닉층 크기 H
    m.OPTIMIZER = "adam"       # "sgd" | "momentum" | "adam"
    m.LR = 0.01
    m.EPOCHS = 300             # 조기 종료가 알아서 멈춰요 (상한)
    m.BATCH_SIZE = 64
    m.SEED = 1234
    m.DEVICE = "auto"          # "auto"=MPS 있으면 사용, 없으면 CPU. "cpu"/"mps" 로 못박기 가능

    # --- v0.2.2 에서 물려받은 정규화 ---
    m.WEIGHT_DECAY = 1e-4
    m.INIT = "kaiming"
    m.USE_BN = False           # 이 데이터엔 손해 → 끔
    m.LABEL_SMOOTHING = 0.1

    # --- ★ v0.2.3 이 더한 세 가지 ---
    m.DROPOUT = 0.1            # 은닉층 뒤 dropout (0.0 = 끔). 0.1~0.3 을 바꿔가며 비교해 보세요
    m.TIE_WEIGHTS = True       # 출력층 ↔ 임베딩 가중치 공유 (HIDDEN == EMBED 필요)
    m.LR_SCHEDULE = "cosine"   # "none" | "cosine" | "step" | "plateau"
    m.LR_MIN = 0.0             # cosine 이 내려갈 바닥
    m.LR_STEP = 50             # step: 몇 에폭마다
    m.LR_GAMMA = 0.5           # step/plateau: 줄이는 비율
    m.LR_PATIENCE = 5          # plateau: 몇 에폭 정체하면 줄일지

    # --- 조기 종료 (v0.1.0 부터 공통) ---
    m.EARLY_STOPPING = True    # 학습 손실·검증 PPL 중 하나라도 개선되면 지속
    m.PATIENCE = 20            # 둘 다 이만큼 정체하면 중단
    m.MIN_DELTA = 0.0
    # ==============================================================

    m.run_train(model.DATA_PATH, model.MODEL_PATH, model.VOCAB_PATH, model.VALID_PATH)
