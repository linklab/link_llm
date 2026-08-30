# -*- coding: utf-8 -*-
"""
test.py  (v0.2.3)  -  일반화·튜닝 모델 평가

- 퍼플렉서티(PPL): 학습/검증
- 이어쓰기(completion)
핵심: v0.2.2(검증 3.48)와 같은 문맥(N=3)에서 dropout·weight tying·LR 스케줄을 더한 버전이에요.
      학습 vs 검증 PPL **격차**가 줄면 일반화 도구가 값을 한 것입니다.

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
    params = sum(p.numel() for p in lm.net.parameters())

    print("=== v0.2.3 (일반화·튜닝 · dropout / weight tying / LR 스케줄) ===\n")
    print(f"학습 데이터({len(train_sents)}문장) PPL : {ppl_train:6.2f}")
    print(f"검증 데이터({len(valid_sents)}문장) PPL : {ppl_valid:6.2f}")
    print(f"학습 vs 검증 격차                : {ppl_valid / ppl_train:6.2f}배   ← 이 버전의 핵심 지표")
    print(f"파라미터                          : {params:,}개"
          f"{'  (weight tying 으로 V×H 절약)' if lm.TIE_WEIGHTS else ''}\n")

    print("→ 격차가 v0.2.2 보다 작아지면 '일반화 도구가 값을 함'.")
    print("  검증 PPL 자체도 v0.2.2(3.48) 와 비교해 보세요.\n")

    print("--- 이어쓰기(completion) 예시 (greedy, temperature=0.0) ---")
    print("    산문으로 '사전학습'한 모델이라 문장 이어쓰기를 봅니다. (대화 능력은 v0.5 SFT에서)")
    for seed in ["아침 일찍", "나는 조용한", "봄 바람"]:
        print(f"  [씨앗] {seed}\n  [생성] {lm.generate(seed, temperature=0.0)}")


if __name__ == "__main__":
    main()
