# -*- coding: utf-8 -*-
"""
test.py  (v0.2.1)  -  N토큰 문맥 임베딩 MLP 평가

- 퍼플렉서티(PPL): 학습/검증 (v0.0.9 와 같은 자로)
- 이어쓰기(completion): 첫머리를 주고 뒤를 이어 써 보기
핵심: 문맥을 앞 N토큰(BLOCK_SIZE)으로 넓혔어요.
      ※ v0.2.0 과의 PPL 차이로 문맥 효과를 판단하면 안 돼요 — 두 버전은 EMBED·HIDDEN·LR 등도
        함께 다릅니다. 문맥만 떼어 보려면 루트의 ablation_block_size.py 를 쓰세요.

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
    print("→ BLOCK_SIZE 를 2·3·4 로 바꿔가며 비교해 보세요 (E·H 는 고정한 채로!).")
    print("  v0.2.0 과의 차이는 문맥 말고 EMBED·HIDDEN·LR 도 달라 문맥 효과가 아니에요 —")
    print("  문맥만 떼어 보려면 루트의 ablation_block_size.py 를 실행하세요.\n")

    print("--- 이어쓰기(completion) 예시 (greedy, temperature=0.0) ---")
    print("    산문으로 '사전학습'한 모델이라 문장 이어쓰기를 봅니다. (대화 능력은 v0.5 SFT에서)")
    for seed in ["아침 일찍", "나는 조용한", "봄 바람"]:
        print(f"  [씨앗] {seed}\n  [생성] {lm.generate(seed, temperature=0.0)}")


if __name__ == "__main__":
    main()
