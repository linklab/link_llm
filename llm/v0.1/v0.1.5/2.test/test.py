# -*- coding: utf-8 -*-
"""
test.py  (v0.1.5)  -  기준선 대결: 카운트 vs 신경망 PPL 비교 (캡스톤)

같은 학습/검증 데이터로 여러 버전의 퍼플렉서티(PPL)를 재서 나란히 보여줘요.
  - v0.0.9 (카운트 bigram)         ← 개수 세기 시대의 기준선
  - v0.1.0 ~ v0.1.3 (신경망, 1토큰 문맥)
  - v0.1.4 (신경망, 2토큰 문맥 = 카운트와 같은 조건)
'검증 PPL' 이 낮을수록 '처음 보는 것'에 강한 거예요 → "신경망이 정말 이겼나?"

(각 버전은 미리 그 폴더의 1.train/train.py 로 학습해 model.json 이 있어야 비교에 포함돼요.
 신경망 버전은 PyTorch 가 필요하고, v0.0.9 카운트 모델은 이미 학습돼 있어요.)
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

VERSIONS = ["v0.0.9", "v0.1.0", "v0.1.1", "v0.1.2", "v0.1.3", "v0.1.4", "v0.1.5"]


def main():
    reader = _model.Model()                       # 문장 읽기용 (torch 불필요)
    train_sents = reader.read_sentences(_model.DATA_PATH)
    valid_sents = reader.read_sentences(_model.VALID_PATH)

    rows, skipped = _model.compare(train_sents, valid_sents, VERSIONS)

    print("=== v0.1.5 기준선 대결 — 카운트 vs 신경망 (PPL) ===")
    print(f"(학습 {len(train_sents)}문장 / 검증 {len(valid_sents)}문장)\n")
    print(f"  {'버전':<8} {'종류':<6} {'학습PPL':>8} {'검증PPL':>8} {'격차':>7} "
          f"{'top-1':>7} {'후보有':>7}")
    print("  " + "-" * 60)
    for v, kind, ppl_tr, ppl_va, top1, cov in rows:
        print(f"  {v:<8} {kind:<6} {ppl_tr:8.2f} {ppl_va:8.2f} {ppl_va/ppl_tr:6.1f}배 "
              f"{top1*100:6.1f}% {cov*100:6.1f}%")

    if skipped:
        print("\n  (제외됨)")
        for v, why in skipped:
            print(f"   - {v}: {why}")

    # 결론: 검증 PPL 이 가장 낮은 게 '일반화 승자' — 그리고 정확도 승자와 다를 수 있어요
    if rows:
        winner = min(rows, key=lambda r: r[3])
        best_acc = max(rows, key=lambda r: r[4])
        base = next((r for r in rows if r[0] == "v0.0.9"), None)
        neural = [r for r in rows if r[1] == "신경망"]
        print(f"\n→ 검증 PPL 최저: {winner[0]} ({winner[1]}) = {winner[3]:.2f}")
        if base and winner[0] != "v0.0.9":
            print(f"  → 신경망이 카운트 기준선(v0.0.9 {base[3]:.2f})을 이겼어요. ✅")
        elif base and winner[0] == "v0.0.9":
            print(f"  → 아직 카운트(v0.0.9)가 더 낮아요. 신경망 하이퍼파라미터를 더 손봐야 해요.")

        print(f"→ top-1 정확도 최고: {best_acc[0]} ({best_acc[1]}) = {best_acc[4]*100:.1f}%")
        if best_acc[0] != winner[0]:
            print("  → PPL 승자와 정확도 승자가 **다르네요**. '확률을 잘 배분하는 것' 과")
            print("     '1등을 잘 맞히는 것' 은 다른 능력이에요 — 지표를 하나만 보면 안 되는 이유.")
        if base and neural:
            print(f"→ '정답이 후보에 있음': 카운트 {base[5]*100:.1f}%  vs  "
                  f"신경망 {max(r[5] for r in neural)*100:.1f}%")
            print("  → 카운트는 표에 없는 조합을 아예 못 만들어요. PPL 에서는 FLOOR 가 가려주는 한계예요.")


if __name__ == "__main__":
    main()
