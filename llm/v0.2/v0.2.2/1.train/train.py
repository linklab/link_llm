# -*- coding: utf-8 -*-
"""
train.py  (v0.2.2)  -  학습 실행기 (+ 하이퍼파라미터)  · 초기화·정규화(BatchNorm)

공용 data/pretrain/train.txt 를 학습해 0.model/model.pt + vocab.json 을 만듭니다.
v0.2.1 대비: INIT="kaiming"(올바른 초기화) + BatchNorm(모델) + weight_decay + 조기 종료로
v0.2.1(학습 3.00 vs 검증 3.37)과 같은 조건에서 초기화·정규화의 효과를 봅니다.

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
    m.BLOCK_SIZE = 3           # 문맥 길이 N (v0.2.1 과 같은 조건에서 개선 확인)
    m.EMBED = 32               # 임베딩 차원 E
    m.HIDDEN = 128             # 은닉층 크기
    m.OPTIMIZER = "adam"       # "sgd" | "momentum" | "adam"
    m.LR = 0.01                # 이 작은 데이터엔 0.05 는 진동 → 0.01 이 안정적
    m.EPOCHS = 1_500           # 상한. 실제 종료 지점은 조기 종료가 정해요
    m.BATCH_SIZE = 64
    m.SEED = 1234
    m.DEVICE = "auto"    # "auto"=애플 실리콘 GPU(MPS) 있으면 사용, 없으면 CPU. "cpu"/"mps" 로 못박기 가능
    m.WEIGHT_DECAY = 1e-4      # ★ 가벼운 L2 정규화 (1e-3 은 이 크기 모델엔 과해 underfit)
    m.INIT = "kaiming"         # ★ tanh 이득 반영 초기화 ("zeros"/"default" 로 비교 가능)
    m.USE_BN = False           # ★ BatchNorm: 이 작은 데이터엔 손해(41 vs 38) → 끔. True 로 비교해 보세요
    m.LABEL_SMOOTHING = 0.1    # 과신 완화

    # 조기 종료: 학습 손실(Loss)이 PATIENCE 에폭 동안 안 좋아지면 멈추고 **최고 가중치**로 복원
    m.EARLY_STOPPING = True
    m.PATIENCE = 20            # 조기 종료 인내 에폭(통일)
    m.MIN_DELTA = 0.0          # 개선으로 인정할 최소 폭(통일)
    # ==============================================================

    m.run_train(model.DATA_PATH, model.MODEL_PATH, model.VOCAB_PATH, model.VALID_PATH)
