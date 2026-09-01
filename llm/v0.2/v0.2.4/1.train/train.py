# -*- coding: utf-8 -*-
"""
train.py  (v0.2.4)  -  학습 실행기 (+ 하이퍼파라미터)  · 임베딩 시각화

공용 data/pretrain/train.txt 를 학습해 0.model/model.pt + vocab.json 을 만듭니다.
※ 이 버전은 **모델을 바꾸지 않아요** — 설정도 v0.2.3 과 똑같습니다.
   v0.2.4 가 더한 건 학습된 임베딩을 들여다보는 **분석 도구**뿐이라,
   학습 결과(검증 PPL 3.26)도 v0.2.3 과 같아야 정상이에요. 분석은 2.test/test.py 에서 합니다.

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
    m.EMBED = 256              # ★ 임베딩 차원 E — weight tying 을 쓰려면 HIDDEN 과 같아야 해요
    m.HIDDEN = 256             # ★ 은닉층 크기 H
    m.OPTIMIZER = "adam"       # "sgd" | "momentum" | "adam"
    m.LR = 0.0003
    m.EPOCHS = 1_500           # ★ cosine 스케줄의 T_max 이기도 해요 — 상한이자 '감쇠 기간'
    m.BATCH_SIZE = 64
    m.SEED = 1234
    m.DEVICE = "auto"          # "auto"=MPS 있으면 사용, 없으면 CPU. "cpu"/"mps" 로 못박기 가능

    # --- v0.2.2 에서 물려받은 정규화 ---
    m.WEIGHT_DECAY = 1e-4
    m.INIT = "kaiming"
    m.USE_BN = False           # 이 데이터엔 손해 → 끔
    m.LABEL_SMOOTHING = 0.1

    # --- v0.2.3 이 더한 튜닝 3종 (v0.2.4 는 모델을 안 바꿔요) ---
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
