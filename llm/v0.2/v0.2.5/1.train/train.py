# -*- coding: utf-8 -*-
"""
train.py  (v0.2.5)  -  학습 실행기 (+ 하이퍼파라미터)  · 3자 비교 캡스톤

공용 data/pretrain/train.txt 를 학습해 0.model/model.pt + vocab.json 을 만듭니다.

※ 이 버전은 **모델을 바꾸지 않아요** — 구조·손실·저장 전부 v0.2.4 그대로입니다.
   딱 하나 다른 건 EPOCHS 예요. 아래 설명대로 **캡스톤이 재현 가능해야 하기 때문**입니다.
   캡스톤이 재는 건 '튜닝을 얼마나 잘했나' 가 아니라 **'토큰을 어떻게 표현하나'** 니까요.

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

    # ====== 하이퍼파라미터 (EPOCHS 만 빼고 v0.2.4 와 동일 — 바꾸면 비교가 깨져요) ======
    m.BLOCK_SIZE = 3           # 문맥 길이 N
    m.EMBED = 128              # 임베딩 차원 E — weight tying 을 쓰려면 HIDDEN 과 같아야 해요
    m.HIDDEN = 128             # 은닉층 크기 H
    m.OPTIMIZER = "adam"       # "sgd" | "momentum" | "adam"
    m.LR = 0.01
    # ★ 여기만 v0.2.4(1500)와 달라요 — v0.2.3 이 진단한 결함을 실제로 고치는 자리입니다.
    #   EPOCHS 는 cosine 의 T_max 를 겸해서 '상한' 이자 '감쇠 기간' 이에요. 1500 으로 두면
    #   학습률이 한참 안 줄어 조기 종료가 언제 걸리느냐가 우연이 되고, 같은 설정 3회가
    #   3.26 / 3.59 / 3.66 으로 흩어집니다(편차 0.40). 감쇠가 예산 안에서 끝나는 300 이면
    #   실측 3.2535 / 3.2511 / 3.2544 — **편차 0.003** 로 잦아들어요(씨드 1234·7·99).
    #   캡스톤은 세 시대의 대표를 공정하게 비교하는 자리라, 임베딩 대표가 운에 좌우되면 안 돼요.
    m.EPOCHS = 300             # cosine 감쇠가 실제로 끝나는 값 (T_max = EPOCHS)
    m.BATCH_SIZE = 64
    m.SEED = 1234              # MPS 는 같은 씨드도 완전 재현은 안 돼요 — 그래서 편차를 줄인 겁니다
    m.DEVICE = "auto"          # "auto"=MPS 있으면 사용, 없으면 CPU

    # --- v0.2.2 에서 물려받은 초기화·정규화 ---
    m.WEIGHT_DECAY = 1e-4
    m.INIT = "kaiming"
    m.USE_BN = False           # 이 데이터엔 손해 → 끔
    m.LABEL_SMOOTHING = 0.1

    # --- v0.2.3 이 더한 튜닝 3종 ---
    m.DROPOUT = 0.1            # 은닉층 뒤 dropout (0.0 = 끔)
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
    # ==========================================================================

    m.run_train(model.DATA_PATH, model.MODEL_PATH, model.VOCAB_PATH, model.VALID_PATH)
