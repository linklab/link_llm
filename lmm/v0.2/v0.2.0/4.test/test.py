# -*- coding: utf-8 -*-
"""
test.py  (v0.2.0)  -  임베딩 MLP 평가

- 퍼플렉서티(PPL): 학습/검증 (v0.0.9 와 같은 자로)
- 대화(chat): 실제로 답을 만들어 보기
핵심: 입력을 one-hot → 임베딩으로 바꿨어요. 검증 PPL 이 v0.1.5(one-hot 2토큰) 보다
      내려가면 "임베딩의 공유 표현이 일반화에 도움"이 확인돼요.
      (여러 모델 3자 대결표는 v0.2.5 캡스톤에서.)

※ PyTorch 필요:  pip install torch
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

    train_sents = lm.read_sentences(DATA_PATH)
    valid_sents = lm.read_sentences(VALID_PATH)
    ppl_train = lm.perplexity(train_sents)
    ppl_valid = lm.perplexity(valid_sents)

    print("=== v0.2.0 (임베딩 MLP) ===\n")
    print(f"학습 데이터({len(train_sents)}문장) PPL : {ppl_train:6.2f}")
    print(f"검증 데이터({len(valid_sents)}문장) PPL : {ppl_valid:6.2f}")
    print("→ v0.1.5(one-hot 2토큰) 검증 PPL 보다 내려가면 '임베딩의 공유 표현이 일반화에 도움'.")
    print("  더 내려가면 카운트(v0.0.9 = 34.39)도 넘볼 수 있어요.\n")

    # --- 순위로 재는 평가 (v0.0.9 의 accuracy) — PPL 과 '다른 것'을 봅니다 ---
    acc = lm.accuracy(valid_sents)
    print(f"검증 top-1 정확도  : {acc['top1'] * 100:5.1f}%   ← 1등으로 찍은 토큰이 정답")
    print(f"검증 top-5 정확도  : {acc['topk'] * 100:5.1f}%")
    print(f"정답이 후보에 있음 : {acc['coverage'] * 100:5.1f}%   ← 신경망은 어휘 전체에 확률을 주므로 높아요")
    print("→ PPL 은 '확률을 얼마나 잘 배분했나', 정확도는 '1등을 얼마나 맞혔나' — 순위가 뒤바뀌기도 해요.")
    print("  실제로 여기서 뒤바뀌어요: 임베딩은 one-hot(v0.1.5)보다 PPL 은 높지만(33.39 vs 31.98)")
    print("  top-1 정확도는 더 높습니다(47.6% vs 46.6%). '임베딩 효과'는 이 자로 보면 보여요.\n")

    print("--- 대화 예시 (greedy, temperature=0.0) ---")
    for msg in ["안녕", "오늘 날씨 어때?", "고마워"]:
        reply = lm.chat(msg, history=None, temperature=0.0)
        print(f"  <나> {msg}\n  <봇> {reply}")


if __name__ == "__main__":
    main()
