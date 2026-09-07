# -*- coding: utf-8 -*-
"""
ablation_tying.py  -  "weight tying 을 포기하고 E≠H 를 쓰는 게 나을까?" 를 재는 실험

[왜 필요한가요?]
v0.2.3 의 weight tying(`fc2.weight = emb.weight`)은 파라미터를 35% 줄여줘요.
그런데 **공짜가 아닙니다** — 두 행렬 모양이 같아야 하니 `HIDDEN == EMBED` 가 **강제**돼요.

그래서 이 저장소는 v0.2.1~2.5 가 전부 E=H=256 이고, **E≠H 조합을 한 번도 안 시험했습니다.**
tying 이 막고 있던 이득이 있는지 확인하는 게 이 실험의 목적이에요.

[실험]  N=3 고정 · dropout·LR 스케줄 끔(도구 효과를 섞지 않으려고)

    A. E=256 H=256  tied     ← 기준: 현재 v0.2.3~2.5
    B. E=256 H=256  untied   ← 묶기만 푼 것 (tying 의 순수 비용)
    C. E=128 H=256  untied   ← ★ 비슷한 크기인데 E≠H
    D. E=256 H=512  untied   ← ★ H 를 키움 (tying 이면 불가능)
    E. E=128 H=512  untied   ← ★ E 는 줄이고 H 는 키움

A 가 알려진 값을 재현하면 나머지를 믿을 수 있어요.
**C 가 핵심** — A 와 크기가 비슷하니, 여기서 A 를 이기면 tying 이 이득을 막고 있던 겁니다.

[읽는 법]  PPL 과 파라미터를 **함께** 보세요. tying 을 포기하려면 그만한 PPL 이득이 있어야 해요.

[실행]
    python3 ablation_tying.py                 # 5개 전부 (MPS 로 40~70분)
    python3 ablation_tying.py --runs A C      # 고른 것만
    python3 ablation_tying.py --json out.json

학습된 모델은 **저장하지 않아요** — 0.model/ 산출물은 그대로입니다.
"""
import os
import sys
import json
import time
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
LM_PATH = os.path.join(HERE, "llm", "v0.2", "v0.2.3", "0.model", "lm.py")   # tying 을 지원
DATA_DIR = os.path.join(HERE, "data", "pretrain")

_spec = importlib.util.spec_from_file_location("llmlm_ablation_tying", LM_PATH)
_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model)

# 모든 실행이 공유하는 고정 설정 (도구는 꺼서 tying/용량 효과만 남깁니다)
BASE = dict(BLOCK_SIZE=3, OPTIMIZER="adam", LR=0.0003, EPOCHS=1500, BATCH_SIZE=64,
            SEED=1234, DEVICE="auto", WEIGHT_DECAY=1e-4, INIT="kaiming", USE_BN=False,
            LABEL_SMOOTHING=0.1, DROPOUT=0.0, LR_SCHEDULE="none",
            EARLY_STOPPING=True, PATIENCE=20, MIN_DELTA=0.0)

RUNS = [
    ("A", 256, 256, True,  "기준 — 현재 v0.2.3~2.5 (tied)"),
    ("B", 256, 256, False, "묶기만 푼 것 (tying 의 순수 비용)"),
    ("C", 128, 256, False, "★ 비슷한 크기인데 E≠H"),
    ("D", 256, 512, False, "★ H 확대 (tying 이면 불가능)"),
    ("E", 128, 512, False, "★ E 축소 + H 확대"),
]


def run_one(tag, embed, hidden, tied, note):
    lm = _model.Model()
    for k, v in BASE.items():
        setattr(lm, k, v)
    lm.EMBED, lm.HIDDEN, lm.TIE_WEIGHTS = embed, hidden, tied

    train_sents = lm.read_sentences(_model.DATA_PATH)
    valid_sents = lm.read_sentences(_model.VALID_PATH)

    print(f"\n{'=' * 72}")
    print(f"[{tag}] {note}")
    print(f"     E={embed} H={hidden} tying={tied}", flush=True)

    started = time.time()
    lm.train(train_sents, valid_sents)
    seconds = time.time() - started

    params = sum(p.numel() for p in lm.net.parameters())
    valid_ppl = lm.perplexity(valid_sents)
    print(f"[{tag}] 완료 {seconds:.0f}s · 채택 epoch {lm.stopped_epoch} · "
          f"파라미터 {params:,} · 검증PPL {valid_ppl:.4f}", flush=True)

    return dict(tag=tag, note=note, embed=embed, hidden=hidden, tied=tied,
                params=params, epoch=lm.stopped_epoch,
                train_ppl=round(lm.perplexity(train_sents), 4),
                valid_ppl=round(valid_ppl, 4), seconds=round(seconds))


def report(results):
    print(f"\n{'=' * 72}")
    print("=== weight tying vs E≠H (N=3 고정 · 도구 끔) ===\n")
    print(f"  {'':<3} {'E':>5} {'H':>5} {'tying':>7} {'검증PPL':>9} {'파라미터':>11} {'기준 대비':>10}")
    print("  " + "-" * 56)
    base = next((r for r in results if r["tag"] == "A"), None)
    for r in results:
        delta = ""
        if base and r is not base:
            dp = r["valid_ppl"] - base["valid_ppl"]
            dn = r["params"] / base["params"]
            delta = f"{dp:+.4f} · {dn:.2f}배"
        print(f"  {r['tag']:<3} {r['embed']:>5} {r['hidden']:>5} {str(r['tied']):>7} "
              f"{r['valid_ppl']:>9.4f} {r['params']:>11,} {delta:>18}")

    if not base:
        print("\n  (A 를 함께 돌려야 비교가 됩니다.)")
        return

    better = [r for r in results if r is not base and r["valid_ppl"] < base["valid_ppl"]]
    print()
    if not better:
        print("  ⇒ **tying 을 포기할 이유가 없어요.** E≠H 조합 중 기준(A)보다 나은 게 없습니다.")
    else:
        b = min(better, key=lambda r: r["valid_ppl"])
        gain = base["valid_ppl"] - b["valid_ppl"]
        cost = b["params"] / base["params"]
        print(f"  ⇒ {b['tag']}(E={b['embed']} H={b['hidden']}) 가 기준보다 PPL {gain:.4f} 낮아요.")
        print(f"    다만 파라미터는 {cost:.2f}배 — 이 이득이 크기를 포기할 만한지 함께 보세요.")
    print("\n  ※ tying 은 PPL 이 아니라 **크기**로 값을 하는 도구예요 (v0.2.3 README 참고).")


def main(argv):
    wanted = None
    if "--runs" in argv:
        wanted = {a.upper() for a in argv[argv.index("--runs") + 1:] if not a.startswith("-")}
    json_path = argv[argv.index("--json") + 1] if "--json" in argv else None

    results = []
    for tag, e, h, tied, note in RUNS:
        if wanted and tag not in wanted:
            continue
        results.append(run_one(tag, e, h, tied, note))
        if json_path:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)

    if results:
        report(results)
    if json_path:
        print(f"\n결과 저장 -> {json_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
