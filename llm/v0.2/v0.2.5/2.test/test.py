# -*- coding: utf-8 -*-
"""
test.py  (v0.2.5)  -  3자 대결 (캡스톤): 카운트 vs one-hot 신경망 vs 임베딩 MLP

검증 자리를 **학습에서 봤는지**로 세 칸에 나눠, 칸마다 따로 PPL 을 재요.
전체 평균 하나로는 안 보이던 것 — "임베딩이 정말 처음 보는 조합에 강한가?" — 를 봅니다.

  ① 본 조합   : 그 문맥에서 그 다음 토큰이 학습에 있었음   (외우기만 해도 맞힘)
  ② 새 조합   : 문맥은 봤지만 그 조합은 처음              (★ 일반화가 필요)
  ③ 새 문맥   : 문맥 자체가 처음                          (★★ 가장 어려움)

(비교할 버전은 각자 1.train/train.py 로 **먼저 학습**해 두어야 표에 나와요.
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

# 세 시대의 완성본 — 각 시대가 '토큰을 어떻게 표현하는가' 를 대표해요
VERSIONS = ["v0.0.9", "v0.1.5", "v0.2.5"]
REPRESENTATION = {
    "v0.0.9": "개수 표 (표현 없음)",
    "v0.1.5": "one-hot V차원 (독립)",
    "v0.2.5": "임베딩 E차원 (공유)",
}


def main():
    reader = _model.Model()                   # 문장 읽기·토크나이즈용 (torch 불필요)
    train_sents = reader.read_sentences(_model.DATA_PATH)
    valid_sents = reader.read_sentences(_model.VALID_PATH)

    print("=== v0.2.5 3자 대결 — 토큰을 어떻게 표현하는가 (캡스톤) ===")
    print(f"(학습 {len(train_sents)}문장 / 검증 {len(valid_sents)}문장 · "
          f"신규성 판정 문맥 {_model.NOVELTY_CONTEXT}토큰)\n")

    rows, skipped = _model.compare(reader, train_sents, valid_sents, VERSIONS)
    if not rows:
        print("  비교할 모델이 없어요.")
        for v, why in skipped:
            print(f"   - {v}: {why}")
        return

    labels = _model.novelty_buckets(reader, train_sents, valid_sents)
    n_all = len(labels)
    share = {b: labels.count(b) / n_all for b in _model.BUCKETS}

    # --- 검증 자리는 어떻게 생겼나 ---
    print("--- 검증 자리 구성 (모델과 무관 · 데이터만 보고 나눔) ---")
    for b in _model.BUCKETS:
        print(f"  {b:<8} {labels.count(b):>6}자리 ({share[b] * 100:4.1f}%)")
    print(f"  {'합계':<8} {n_all:>6}자리\n")

    # --- 대결표 ---
    print("--- 칸별 검증 PPL (낮을수록 좋음) ---")
    print(f"  {'버전':<8} {'표현':<20} {'전체':>7} {'①본 조합':>9} "
          f"{'②새 조합':>9} {'③새 문맥':>9} {'파라미터':>10}")
    print("  " + "-" * 80)
    for r in rows:
        b = r["buckets"]
        print(f"  {r['v']:<8} {REPRESENTATION.get(r['v'], r['kind']):<20} {r['ppl']:7.2f} "
              f"{b[_model.SEEN_PAIR][0]:9.2f} {b[_model.NEW_PAIR][0]:9.2f} "
              f"{b[_model.NEW_CONTEXT][0]:9.2f} {r['params']:10,}")

    if skipped:
        print("\n  (제외됨)")
        for v, why in skipped:
            print(f"   - {v}: {why}")

    # --- 읽는 법 (숫자에서 바로 뽑아요) ---
    print("\n--- 읽는 법 ---")
    best_all = min(rows, key=lambda r: r["ppl"])
    print(f"① 전체 PPL 1등: {best_all['v']} ({best_all['kind']}) = {best_all['ppl']:.2f}")

    for bucket, tag in ((_model.SEEN_PAIR, "외운 자리"),
                        (_model.NEW_PAIR, "일반화가 필요한 자리"),
                        (_model.NEW_CONTEXT, "가장 어려운 자리")):
        best = min(rows, key=lambda r: r["buckets"][bucket][0])
        worst = max(rows, key=lambda r: r["buckets"][bucket][0])
        print(f"   · {bucket}({tag}) 1등: {best['v']} {best['buckets'][bucket][0]:.2f}"
              f"   / 꼴찌: {worst['v']} {worst['buckets'][bucket][0]:.2f}")

    count_row = next((r for r in rows if r["kind"] == "카운트"), None)
    emb_row = next((r for r in rows if r["kind"] == "임베딩"), None)
    onehot_row = next((r for r in rows if r["kind"] == "one-hot"), None)

    if count_row and emb_row:
        print("\n② 카운트 vs 임베딩 — 칸마다 격차가 어떻게 달라지나")
        for b in _model.BUCKETS:
            d = count_row["buckets"][b][0] - emb_row["buckets"][b][0]
            who = "임베딩이 우세" if d > 0 else "카운트가 우세"
            print(f"   · {b:<8} 카운트 {count_row['buckets'][b][0]:7.2f} vs "
                  f"임베딩 {emb_row['buckets'][b][0]:7.2f}  → {who} ({abs(d):.2f})")
        print("   → 격차가 ①에서 ③으로 갈수록 커지면 '임베딩은 처음 보는 것에 강하다'가 맞는 거예요.")

    if onehot_row and emb_row:
        print("\n③ one-hot vs 임베딩 — 같은 신경망, 표현만 다름")
        for b in _model.BUCKETS:
            print(f"   · {b:<8} one-hot {onehot_row['buckets'][b][0]:7.2f} vs "
                  f"임베딩 {emb_row['buckets'][b][0]:7.2f}")
        ratio = onehot_row["params"] / emb_row["params"]
        print(f"   → 파라미터: one-hot {onehot_row['params']:,} vs "
              f"임베딩 {emb_row['params']:,} ({ratio:.1f}배)")

    print("\n④ '정답이 후보에 있기라도 한가' (카운트의 근본 한계)")
    for r in rows:
        print(f"   · {r['v']:<8} {r['coverage'] * 100:5.1f}%   (top-1 {r['top1'] * 100:.1f}%)")
    print("   → 카운트는 표에 없는 조합을 **아예 만들지 못해요.** PPL 에서는 FLOOR 가 가려주는 한계예요.")


if __name__ == "__main__":
    main()
