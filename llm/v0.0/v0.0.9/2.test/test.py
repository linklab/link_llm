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

    train_sentences = lm.read_sentences(DATA_PATH)
    valid_sentences = lm.read_sentences(VALID_PATH)

    ppl_train = lm.perplexity(train_sentences)
    ppl_valid = lm.perplexity(valid_sentences)

    acc = lm.accuracy(valid_sentences)

    print("=== v0.0.9 (퍼플렉서티 평가) ===\n")
    print(f"학습 데이터({len(train_sentences)}문장) PPL : {ppl_train:6.2f}   ← 외운 것이라 낮음")
    print(f"검증 데이터({len(valid_sentences)}문장) PPL : {ppl_valid:6.2f}   ← 처음 보는 것이라 높음")
    print()
    print("→ PPL(퍼플렉서티): '다음 토큰을 얼마나 잘 맞히나' 를 하나의 숫자로 (낮을수록 좋음).")
    print(f"→ 검증이 학습보다 약 {ppl_valid / ppl_train:.1f}배 높아요 = 이게 '외우기 vs 일반화' 격차예요.")
    print("   (개수 세기 모델은 본 것만 잘 맞혀요. 나중에 신경망이 이 PPL 을 낮출 수 있는지 비교합니다.)")
    print()
    print(f"--- 순위로 재는 평가 (검증 {acc['n']}개 자리) ---")
    print(f"top-1 정확도       : {acc['top1'] * 100:5.1f}%   ← 1등으로 찍은 토큰이 정답")
    print(f"top-{acc['k']} 정확도       : {acc['topk'] * 100:5.1f}%")
    print(f"정답이 후보에 있음 : {acc['coverage'] * 100:5.1f}%   ← 나머지는 표에 아예 없어서 못 맞혀요")
    print()
    print("→ PPL 은 '확률을 얼마나 잘 배분했나', 정확도는 '1등을 얼마나 맞혔나' 예요 — 다른 질문이라")
    print("  두 지표의 순위가 뒤바뀌기도 해요.")
    print("→ '정답이 후보에 있음' 이 개수 세기 모델의 한계를 그대로 보여줘요. 표에 없는 조합은")
    print(f"   확률을 줄 방법 자체가 없어서, {100 - acc['coverage'] * 100:.0f}% 는 아무리 잘 골라도 틀립니다.")
    print("   (PPL 에서는 FLOOR 가 대신 메워줘서 이 한계가 잘 안 보여요. 신경망은 softmax 라")
    print("    어휘 전체에 확률이 퍼져서 이 숫자가 훨씬 높아집니다 — v0.1.x 에서 비교해 보세요.)")


if __name__ == "__main__":
    main()
