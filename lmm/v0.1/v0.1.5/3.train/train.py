# -*- coding: utf-8 -*-
"""
train.py  (v0.1.5)  -  학습 실행기 (+ 하이퍼파라미터)

모델·학습은 v0.1.4(2토큰 문맥) 그대로예요. 하이퍼파라미터를 여기서 정해 모델에 넘기고,
1.data/data.txt 를 학습해 2.models/model.json 을 만듭니다.
(이 model.json 이 4.test/test.py 의 '대결'에서 v0.1.5 쪽으로 쓰여요.)

※ PyTorch 필요:  pip install torch
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "lmmlm_" + os.path.basename(_HERE).replace(".", "_"),
    os.path.join(_HERE, "2.models", "lm.py"),
)
model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model)

if __name__ == "__main__":
    m = model.Model()

    # ================= 하이퍼파라미터 (여기서 조정) =================
    m.OPTIMIZER = "adam"       # "sgd" | "momentum" | "adam"
    m.LR = 0.05                # Adam 기준값. SGD/momentum 은 크게(1~10)
    m.EPOCHS = 300
    m.BATCH_SIZE = 64
    m.SEED = 1234
    m.WEIGHT_DECAY = 0.0       # 이 모델들엔 1e-4 에도 underfit 할 만큼 민감 → 0
    m.INIT = "zeros"           # "zeros" | "default"
    m.LABEL_SMOOTHING = 0.2    # ★ 2토큰은 용량이 2배라 과적합↑ → 세게(0.2)
    # ==============================================================

    m.run_train(model.DATA_PATH, model.MODEL_PATH)
