# -*- coding: utf-8 -*-
"""
train.py  (v0.2.1)  -  학습 실행기 (+ 하이퍼파라미터)  · N토큰 문맥 임베딩 MLP

공용 data/pretrain/train.txt 를 학습해 0.model/model.pt(가중치) + vocab.json(어휘) 을 만듭니다.
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

    # ============ 하이퍼파라미터 — v0.2.0 과 BLOCK_SIZE 하나만 다릅니다 ============
    # 이 버전의 개념은 '문맥 확장' 이에요. 그래서 나머지 값은 **v0.2.0 과 글자 하나까지 동일**하게
    # 둡니다. 하나라도 같이 바꾸면 "문맥을 넓혀서 좋아졌다/나빠졌다" 를 말할 수 없어요.
    m.BLOCK_SIZE = 3           # ★ 이 버전이 바꾸는 유일한 값 (2 면 v0.2.0 과 완전히 동일)
    m.EMBED = 256              # v0.2.0 과 동일 — 용량을 건드리면 문맥 효과와 섞여요
    m.HIDDEN = 256             # v0.2.0 과 동일
    m.OPTIMIZER = "adam"       # "sgd" | "momentum" | "adam"
    m.LR = 0.0003              # v0.2.0 과 동일 (Adam 은 1e-4~1e-3 대)
    m.EPOCHS = 1_500           # 상한. 실제 종료 지점은 조기 종료가 정해요
    m.BATCH_SIZE = 64
    m.SEED = 1234
    m.DEVICE = "auto"          # "auto"=MPS 있으면 사용, 없으면 CPU
    m.WEIGHT_DECAY = 0.0       # v0.2.0 과 동일 (정규화 도구는 v0.2.2 에서 도입)
    m.INIT = "zeros"           # v0.2.0 과 동일
    m.LABEL_SMOOTHING = 0.0    # v0.2.0 과 동일
    # sweep 제안: BLOCK_SIZE ∈ {2, 3, 4} — **용량은 고정한 채** 문맥만 바꿔 보세요
    # --- 조기 종료 (v0.1.0 부터 공통) ---
    m.EARLY_STOPPING = True    # ★ 학습 손실(Loss) 기준 조기 종료(최저 손실 가중치 복원)
    m.PATIENCE = 20            # 조기 종료 인내 에폭(통일)
    m.MIN_DELTA = 0.0          # 개선으로 인정할 최소 폭(통일)
    # sweep 제안: BLOCK_SIZE ∈ {2, 3, 4}, EMBED ∈ {16, 32, 64}
    # ==============================================================

    m.run_train(model.DATA_PATH, model.MODEL_PATH, model.VOCAB_PATH, model.VALID_PATH)
