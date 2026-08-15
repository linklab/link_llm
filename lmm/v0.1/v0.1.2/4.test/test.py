# -*- coding: utf-8 -*-
"""
test.py  (v0.1.2)  -  torch.optim 으로 학습한 신경망 bigram 평가

- 퍼플렉서티(PPL): 학습/검증 (v0.0.9 와 같은 자로) → v0.1.0/v0.1.1 과 비슷하면 성공
- 대화(chat): 실제로 답을 만들어 보기
(옵티마이저는 2.models/lm.py 의 OPTIMIZER = "sgd"|"momentum"|"adam" 로 바꿔가며 비교해 보세요.)

※ PyTorch 필요:  pip install torch
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "lmmlm_" + os.path.basename(_HERE).replace(".", "_"),
    os.path.join(_HERE, "2.models", "lm.py"),
)
_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model)
Model = _model.Model
DATA_PATH = _model.DATA_PATH
VALID_PATH = _model.VALID_PATH
MODEL_PATH = _model.MODEL_PATH


def main():
    lm = Model.load_or_exit(MODEL_PATH)

    train_sents = lm.read_sentences(DATA_PATH)
    valid_sents = lm.read_sentences(VALID_PATH)
    ppl_train = lm.perplexity(train_sents)
    ppl_valid = lm.perplexity(valid_sents)

    print("=== v0.1.2 (torch.optim 옵티마이저) ===\n")
    print(f"학습 데이터({len(train_sents)}문장) PPL : {ppl_train:6.2f}")
    print(f"검증 데이터({len(valid_sents)}문장) PPL : {ppl_valid:6.2f}")
    print("→ v0.1.0/v0.1.1 과 PPL 이 비슷하면, '갱신 방법만 바꿔도 결과는 같다'가 확인돼요.")
    print("  (옵티마이저별 수렴 속도 차이는 학습 로그의 손실 곡선에서 확인.)\n")

    print("--- 대화 예시 (greedy, temperature=0.0) ---")
    for msg in ["안녕", "오늘 날씨 어때?", "고마워"]:
        reply = lm.chat(msg, history=None, temperature=0.0)
        print(f"  <나> {msg}\n  <봇> {reply}")


if __name__ == "__main__":
    main()
