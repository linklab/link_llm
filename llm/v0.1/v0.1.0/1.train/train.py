# -*- coding: utf-8 -*-
"""
train.py  (v0.1.0)  -  학습 실행기 (+ 하이퍼파라미터)

하이퍼파라미터를 여기서 정해 모델에 넘기고, 공용 data/data.txt 를 학습해
0.model/model.json 을 만듭니다.

※ 이 버전부터 PyTorch 가 필요합니다:  pip install torch
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
    m.HIDDEN = 128       # 은닉층 크기 (2층 MLP)
    m.LR = 1.0           # 학습률 (2층 MLP 는 예전 선형(10)보다 작게)
    m.EPOCHS = 1_500    # 전체 데이터를 몇 번 반복할지
    m.SEED = 1234        # 초기 가중치를 재현 가능하게
    # --- 조기 종료 (early stopping) ---
    # 검증 PPL 이 PATIENCE 에폭 동안 나아지지 않으면 멈추고, **가장 좋았던 가중치**로 되돌려요.
    # 그래서 EPOCHS 는 이제 '정확히 맞춰야 하는 값'이 아니라 넉넉한 **상한**이면 됩니다.
    m.EARLY_STOPPING = True    # False 면 EPOCHS 를 끝까지 돕니다
    m.PATIENCE = 100           # full-batch 는 에폭당 갱신이 1번뿐이라 아주 느리게 내려가요 → 크게
    m.MIN_DELTA = 0.0          # 이만큼은 좋아져야 '개선'으로 인정
    # ==============================================================

    m.run_train(model.DATA_PATH, model.MODEL_PATH, model.VOCAB_PATH, model.VALID_PATH)
