# -*- coding: utf-8 -*-
"""
test.py  (v0.2.2)  -  초기화·정규화(BatchNorm) 모델 평가

- 퍼플렉서티(PPL): 학습/검증
- 이어쓰기(completion)
핵심: v0.2.1(검증 3.37)과 같은 용량·같은 문맥에서 초기화·정규화만 더한 버전이에요.
      검증 PPL 이 내려가면 그 도구들이 이 규모에서도 값을 한다는 뜻입니다. (실측 3.48)

※ PyTorch 필요:  pip install torch
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "llmlm_" + os.path.basename(_HERE).replace(".", "_"),
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

    print("=== v0.2.2 (초기화·정규화 · BatchNorm) ===\n")
    print(f"학습 데이터({len(train_sents)}문장) PPL : {ppl_train:6.2f}")
    print(f"검증 데이터({len(valid_sents)}문장) PPL : {ppl_valid:6.2f}")
    print("→ v0.2.1(검증 3.37) 보다 내려가면 '초기화·정규화가 이 규모에서도 값을 함'.")
    print("  카운트(2.96)·임베딩 v0.2.0(2.94) 과도 비교해 보세요. (실측 3.48)\n")

    print("--- 이어쓰기(completion) 예시 (greedy, temperature=0.0) ---")
    print("    산문으로 '사전학습'한 모델이라 문장 이어쓰기를 봅니다. (대화 능력은 v0.5 SFT에서)")
    for seed in ["아침 일찍", "나는 조용한", "봄 바람"]:
        print(f"  [씨앗] {seed}\n  [생성] {lm.generate(seed, temperature=0.0)}")


if __name__ == "__main__":
    main()
