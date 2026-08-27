# -*- coding: utf-8 -*-
"""
train.py  (v0.2.1)  -  학습 실행기 (+ 하이퍼파라미터)  · N토큰 문맥 임베딩 MLP

공용 data/data.txt 를 학습해 0.model/model.pt(가중치) + vocab.json(어휘) 을 만듭니다.
v0.2.0 대비 새 하이퍼파라미터: BLOCK_SIZE (문맥 길이 N = 앞 몇 토큰을 볼지).

※ PyTorch 필요:  pip install torch
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "lmmlm_" + os.path.basename(_HERE).replace(".", "_"),
    os.path.join(_HERE, "0.model", "lm.py"),
)
model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model)

if __name__ == "__main__":
    m = model.Model()

    # ================= 하이퍼파라미터 (여기서 조정) =================
    m.BLOCK_SIZE = 3           # ★ 문맥 길이 N = 앞 몇 토큰을 볼지 (2면 v0.2.0 과 동일)
    m.EMBED = 32               # 임베딩 차원 E
    m.HIDDEN = 128             # 은닉층 크기 (2층 MLP)
    m.OPTIMIZER = "adam"       # "sgd" | "momentum" | "adam"
    m.LR = 0.01                # ★ 0.05 는 이 깊이(emb→fc1→fc2)에 과해 진동 → 0.01 로 안정화
    m.EPOCHS = 1_500
    m.BATCH_SIZE = 64
    m.SEED = 1234
    m.WEIGHT_DECAY = 0.0       # 임베딩 병목이 정규화 역할 → 우선 0
    m.INIT = "default"         # ★ "zeros"(fc2=0)는 Adam 과 궁합이 나빠 미수렴 → 무작위 초기화
    m.LABEL_SMOOTHING = 0.1    # 과신 완화
    m.EARLY_STOPPING = True    # ★ 검증 PPL 기준 조기 종료(최고 가중치 복원) — 과적합 방지
    m.PATIENCE = 15
    # sweep 제안: BLOCK_SIZE ∈ {2, 3, 4}, EMBED ∈ {16, 32, 64}
    # ==============================================================

    m.run_train(model.DATA_PATH, model.MODEL_PATH, model.VOCAB_PATH, model.VALID_PATH)
