# -*- coding: utf-8 -*-
"""
test.py  (v0.1.0)  -  신경망 bigram 평가

이 버전이 '완결'인지 확인해요:
  1) 퍼플렉서티(PPL) — 학습/검증 데이터에서 (v0.0.9 와 '같은 자'로 잽니다)
  2) 대화(chat) — 실제로 답을 만들어 보기

※ PyTorch 가 필요합니다:  pip install torch
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

    # 1) 퍼플렉서티 (v0.0.9 의 perplexity() 를 그대로 물려받아 사용)
    train_sents = lm.read_sentences(DATA_PATH)
    valid_sents = lm.read_sentences(VALID_PATH)
    ppl_train = lm.perplexity(train_sents)
    ppl_valid = lm.perplexity(valid_sents)

    print("=== v0.1.0 (신경망 bigram + autograd) ===\n")
    print(f"학습 데이터({len(train_sents)}문장) PPL : {ppl_train:6.2f}")
    print(f"검증 데이터({len(valid_sents)}문장) PPL : {ppl_valid:6.2f}")
    print("→ v0.0.9(개수 bigram)의 PPL 과 비교해 보세요. 신경망이 '같은 문맥 길이'로")
    print("  얼마나 비슷/다른지 확인하는 게 이 버전의 핵심이에요. (본격 비교는 v0.1.4)\n")

    # 2) 대화 — 실제로 답을 만들어 보기 (완결성 확인)
    print("--- 대화 예시 (greedy, temperature=0.0) ---")
    for msg in ["안녕", "오늘 날씨 어때?", "고마워"]:
        reply = lm.chat(msg, history=None, temperature=0.0)
        print(f"  <나> {msg}\n  <봇> {reply}")


if __name__ == "__main__":
    main()
