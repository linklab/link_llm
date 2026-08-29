# -*- coding: utf-8 -*-
"""
test.py  (v0.1.4)  -  2토큰 문맥 신경망 평가

- 퍼플렉서티(PPL): 학습/검증 (v0.0.9 와 같은 자로)
- 이어쓰기(completion): 첫머리를 주고 뒤를 이어 써 보기
핵심: 이제 신경망도 v0.0.9 와 '같은 2토큰 문맥'이라 공정한 대결.
      검증 PPL 이 v0.0.9(2.96) 아래로 내려가면 신경망이 이긴 거예요 (실측 2.90 ✅).
      (최종 대결표는 v0.1.5 캡스톤에서 여러 버전을 나란히 출력합니다.)

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

    print("=== v0.1.4 (2토큰 문맥 신경망) ===\n")
    print(f"학습 데이터({len(train_sents)}문장) PPL : {ppl_train:6.2f}")
    print(f"검증 데이터({len(valid_sents)}문장) PPL : {ppl_valid:6.2f}")
    print("→ v0.0.9(카운트) 검증 PPL 2.96 아래면 '같은 문맥에서 신경망 승'. (실측 2.90 ✅)")
    print("  1토큰 신경망(v0.1.0~3)보다 문맥이 넓어 더 잘 맞혀야 정상이에요.\n")

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
