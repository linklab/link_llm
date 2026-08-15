# -*- coding: utf-8 -*-
"""
test.py  (v0.1.3)  -  정규화·초기화 후 '일반화' 평가

핵심은 **학습 PPL vs 검증 PPL 격차**예요:
  - v0.0.9(개수): 학습 2.79 vs 검증 34.39 (약 12배)
  - v0.1.3: weight_decay/초기화로 이 격차가 줄어드는지(검증 PPL↓) 확인

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

    print("=== v0.1.3 (정규화 · 초기화) ===\n")
    print(f"학습 데이터({len(train_sents)}문장) PPL : {ppl_train:6.2f}")
    print(f"검증 데이터({len(valid_sents)}문장) PPL : {ppl_valid:6.2f}")
    print(f"→ 격차(검증/학습): {ppl_valid / ppl_train:.1f}배  "
          f"(참고: v0.0.9 개수 모델은 약 12배)")
    print("  weight_decay/초기화로 이 격차가 줄면 '일반화 개선'이 확인돼요.\n")

    print("--- 대화 예시 (greedy, temperature=0.0) ---")
    for msg in ["안녕", "오늘 날씨 어때?", "고마워"]:
        reply = lm.chat(msg, history=None, temperature=0.0)
        print(f"  <나> {msg}\n  <봇> {reply}")


if __name__ == "__main__":
    main()
