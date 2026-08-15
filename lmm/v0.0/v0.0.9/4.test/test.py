# -*- coding: utf-8 -*-
"""
test.py  (v0.0.9)  -  퍼플렉서티(perplexity) 평가

학습 데이터와 '처음 보는' 검증 데이터의 PPL 을 각각 재서,
모델이 얼마나 잘 맞히는지 + '외우기 vs 일반화' 격차를 봅니다.
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

    train_sentences = lm.read_sentences(DATA_PATH)
    valid_sentences = lm.read_sentences(VALID_PATH)

    ppl_train = lm.perplexity(train_sentences)
    ppl_valid = lm.perplexity(valid_sentences)

    print("=== v0.0.9 (퍼플렉서티 평가) ===\n")
    print(f"학습 데이터({len(train_sentences)}문장) PPL : {ppl_train:6.2f}   ← 외운 것이라 낮음")
    print(f"검증 데이터({len(valid_sentences)}문장) PPL : {ppl_valid:6.2f}   ← 처음 보는 것이라 높음")
    print()
    print("→ PPL(퍼플렉서티): '다음 토큰을 얼마나 잘 맞히나' 를 하나의 숫자로 (낮을수록 좋음).")
    print(f"→ 검증이 학습보다 약 {ppl_valid / ppl_train:.1f}배 높아요 = 이게 '외우기 vs 일반화' 격차예요.")
    print("   (개수 세기 모델은 본 것만 잘 맞혀요. 나중에 신경망이 이 PPL 을 낮출 수 있는지 비교합니다.)")


if __name__ == "__main__":
    main()
