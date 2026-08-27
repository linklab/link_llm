# -*- coding: utf-8 -*-
"""
test.py  (v0.2.1)  -  N토큰 문맥 임베딩 MLP 평가

- 퍼플렉서티(PPL): 학습/검증 (v0.0.9 와 같은 자로)
- 대화(chat): 실제로 답을 만들어 보기
핵심: 문맥을 앞 N토큰(BLOCK_SIZE)으로 넓혔어요. 검증 PPL 이 v0.2.0(2토큰) 보다
      내려가면 "문맥을 더 보니 일반화에 도움"이 확인돼요.

※ PyTorch 필요:  pip install torch
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "lmmlm_" + os.path.basename(_HERE).replace(".", "_"),
    os.path.join(_HERE, "0.model", "lm.py"),
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

    N = getattr(lm, "BLOCK_SIZE", "?")
    print(f"=== v0.2.1 (앞 {N}토큰 문맥 임베딩 MLP) ===\n")
    print(f"학습 데이터({len(train_sents)}문장) PPL : {ppl_train:6.2f}")
    print(f"검증 데이터({len(valid_sents)}문장) PPL : {ppl_valid:6.2f}")
    print("→ v0.2.0(2토큰) 보다 검증 PPL 이 내려가면 '문맥 확장이 일반화에 도움'.")
    print("  BLOCK_SIZE 를 2·3·4 로 바꿔가며 비교해 보세요.\n")

    print("--- 대화 예시 (greedy, temperature=0.0) ---")
    for msg in ["안녕", "오늘 날씨 어때?", "고마워"]:
        reply = lm.chat(msg, history=None, temperature=0.0)
        print(f"  <나> {msg}\n  <봇> {reply}")


if __name__ == "__main__":
    main()
