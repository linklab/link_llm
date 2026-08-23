# -*- coding: utf-8 -*-
"""
test.py  (v0.1.4)  -  2토큰 문맥 신경망 평가

- 퍼플렉서티(PPL): 학습/검증 (v0.0.9 와 같은 자로)
- 대화(chat): 실제로 답을 만들어 보기
핵심: 이제 신경망도 v0.0.9 와 '같은 2토큰 문맥'이라 공정한 대결.
      검증 PPL 이 v0.0.9(34.39) 아래로 내려가면 신경망이 이긴 거예요.
      (최종 대결표는 v0.1.5 캡스톤에서 여러 버전을 나란히 출력합니다.)

※ PyTorch 필요:  pip install torch
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "llmlm_" + os.path.basename(_HERE).replace(".", "_"),
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

    print("=== v0.1.4 (2토큰 문맥 신경망) ===\n")
    print(f"학습 데이터({len(train_sents)}문장) PPL : {ppl_train:6.2f}")
    print(f"검증 데이터({len(valid_sents)}문장) PPL : {ppl_valid:6.2f}")
    print("→ v0.0.9(카운트) 검증 PPL 34.39 아래면 '같은 문맥에서 신경망 승'.")
    print("  1토큰 신경망(v0.1.0~3)보다 문맥이 넓어 더 잘 맞혀야 정상이에요.\n")

    # --- 순위로 재는 평가 (v0.0.9 의 accuracy) — PPL 과 '다른 것'을 봅니다 ---
    acc = lm.accuracy(valid_sents)
    print(f"검증 top-1 정확도  : {acc['top1'] * 100:5.1f}%   ← 1등으로 찍은 토큰이 정답")
    print(f"검증 top-5 정확도  : {acc['topk'] * 100:5.1f}%")
    print(f"정답이 후보에 있음 : {acc['coverage'] * 100:5.1f}%   ← 신경망은 어휘 전체에 확률을 주므로 높아요")
    print("→ PPL 은 '확률을 얼마나 잘 배분했나', 정확도는 '1등을 얼마나 맞혔나' — 순위가 뒤바뀌기도 해요.\n")

    print("--- 대화 예시 (greedy, temperature=0.0) ---")
    for msg in ["안녕", "오늘 날씨 어때?", "고마워"]:
        reply = lm.chat(msg, history=None, temperature=0.0)
        print(f"  <나> {msg}\n  <봇> {reply}")


if __name__ == "__main__":
    main()
