# -*- coding: utf-8 -*-
"""
test.py  (v0.1.3)  -  정규화·초기화 후 '일반화' 평가

핵심은 **학습 PPL vs 검증 PPL 격차**예요:
  - v0.0.9(개수): 학습 2.49 vs 검증 2.96 (약 1.2배)
  - v0.1.3: weight_decay/초기화로 이 격차가 줄어드는지(검증 PPL↓) 확인

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

    print("=== v0.1.3 (정규화 · 초기화) ===\n")
    print(f"학습 데이터({len(train_sents)}문장) PPL : {ppl_train:6.2f}")
    print(f"검증 데이터({len(valid_sents)}문장) PPL : {ppl_valid:6.2f}")
    print(f"→ 격차(검증/학습): {ppl_valid / ppl_train:.1f}배  "
          f"(참고: v0.0.9 개수 모델은 약 12배)")
    print("  weight_decay/초기화로 이 격차가 줄면 '일반화 개선'이 확인돼요.\n")

    # --- 순위로 재는 평가 (v0.0.9 의 accuracy) — PPL 과 '다른 것'을 봅니다 ---
    acc = lm.accuracy(valid_sents)
    print(f"검증 top-1 정확도  : {acc['top1'] * 100:5.1f}%   ← 1등으로 찍은 토큰이 정답")
    print(f"검증 top-5 정확도  : {acc['topk'] * 100:5.1f}%")
    print(f"정답이 후보에 있음 : {acc['coverage'] * 100:5.1f}%   ← 신경망은 어휘 전체에 확률을 주므로 높아요")
    print("→ PPL 은 '확률을 얼마나 잘 배분했나', 정확도는 '1등을 얼마나 맞혔나' — 순위가 뒤바뀌기도 해요.\n")

    print("--- 이어쓰기(completion) 예시 (greedy, temperature=0.0) ---")
    print("    산문으로 '사전학습'한 모델이라 문장 이어쓰기를 봅니다. (대화 능력은 v0.5 SFT에서)")
    for seed in ["아침 일찍", "나는 조용한", "봄 바람"]:
        print(f"  [씨앗] {seed}\n  [생성] {lm.generate(seed, temperature=0.0)}")


if __name__ == "__main__":
    main()
