# -*- coding: utf-8 -*-
"""
sweep.py  (v0.2.3)  -  세 도구(dropout · weight tying · LR 스케줄)를 **하나씩** 켜보는 실험

[왜 필요한가]
v0.2.3 은 도구를 세 개 한꺼번에 켰어요. 그 상태로 v0.2.2 와 비교하면 **무엇이 기여했는지 말할 수 없습니다.**
(루트 `ablation_block_size.py` 가 v0.2.1 에서 같은 함정을 보여줬죠.)
그래서 여기서는 나머지를 고정한 채 **한 번에 하나만** 켜서 각 도구의 몫을 따로 잽니다.

[실험]  E=H=256 · N=3 고정 (v0.2.3 본체와 같은 용량), 조기 종료·weight_decay·label smoothing 은 v0.2.2 와 동일

    ① none      : 도구 셋 다 끔                    ← 기준선
    ② dropout   : dropout 만 켬
    ③ tying     : weight tying 만 켬
    ④ cosine    : LR 스케줄만 켬
    ⑤ all       : 셋 다 켬                          ← v0.2.3 기본 설정

※ tying 을 켜면 파라미터가 V×H 만큼 줄어요. 그래서 표에 **파라미터 수를 같이 찍습니다** —
  PPL 만 보고 "좋아졌다" 하면 크기가 달라진 걸 놓치게 되거든요.

[실행]
    python3 1.train/sweep.py                # 5개 전부 (MPS 로 20~40분)
    python3 1.train/sweep.py --runs all none  # 고른 것만
    python3 1.train/sweep.py --json out.json

학습된 모델은 **저장하지 않아요** — 0.model/ 산출물은 그대로입니다.
"""
import os
import sys
import json
import time
import importlib.util

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # v0.2.3 폴더
_spec = importlib.util.spec_from_file_location("llmlm_sweep_v0_2_3",
                                               os.path.join(_HERE, "0.model", "lm.py"))
_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model)

# 모든 실행이 공유하는 고정 설정 (v0.2.2 와 같은 조건 + E=H 로 맞춤)
BASE = dict(BLOCK_SIZE=3, EMBED=256, HIDDEN=256, OPTIMIZER="adam", LR=0.0003,
            EPOCHS=1500, BATCH_SIZE=64, SEED=1234, DEVICE="auto",
            WEIGHT_DECAY=1e-4, INIT="kaiming", USE_BN=False, LABEL_SMOOTHING=0.1,
            EARLY_STOPPING=True, PATIENCE=20, MIN_DELTA=0.0)

# (이름, 켤 도구) — 나머지는 모두 끈 상태
RUNS = [
    ("none",    dict(DROPOUT=0.0, TIE_WEIGHTS=False, LR_SCHEDULE="none"),   "기준선 — 도구 셋 다 끔"),
    ("dropout", dict(DROPOUT=0.1, TIE_WEIGHTS=False, LR_SCHEDULE="none"),   "dropout 만"),
    ("tying",   dict(DROPOUT=0.0, TIE_WEIGHTS=True,  LR_SCHEDULE="none"),   "weight tying 만"),
    ("cosine",  dict(DROPOUT=0.0, TIE_WEIGHTS=False, LR_SCHEDULE="cosine"), "LR 스케줄만"),
    ("all",     dict(DROPOUT=0.1, TIE_WEIGHTS=True,  LR_SCHEDULE="cosine"), "셋 다 (v0.2.3 기본)"),
]


def run_one(tag, tools, note):
    lm = _model.Model()
    for k, v in {**BASE, **tools}.items():
        setattr(lm, k, v)

    train_sents = lm.read_sentences(_model.DATA_PATH)
    valid_sents = lm.read_sentences(_model.VALID_PATH)

    print(f"\n{'=' * 72}")
    print(f"[{tag}] {note}")
    print(f"     dropout={lm.DROPOUT} tying={lm.TIE_WEIGHTS} schedule={lm.LR_SCHEDULE}", flush=True)

    started = time.time()
    lm.train(train_sents, valid_sents)
    seconds = time.time() - started

    params = sum(p.numel() for p in lm.net.parameters())
    train_ppl = lm.perplexity(train_sents)
    valid_ppl = lm.perplexity(valid_sents)
    print(f"[{tag}] 완료 {seconds:.0f}s · 채택 epoch {lm.stopped_epoch} · 파라미터 {params:,} · "
          f"학습PPL {train_ppl:.4f} · 검증PPL {valid_ppl:.4f}", flush=True)

    return dict(tag=tag, note=note, dropout=lm.DROPOUT, tying=lm.TIE_WEIGHTS,
                schedule=lm.LR_SCHEDULE, params=params, epoch=lm.stopped_epoch,
                train_ppl=round(train_ppl, 4), valid_ppl=round(valid_ppl, 4),
                seconds=round(seconds))


def report(results):
    print(f"\n{'=' * 72}")
    print(f"=== 도구별 몫 (E=H={BASE['EMBED']} · N={BASE['BLOCK_SIZE']} 고정) ===\n")
    print(f"  {'실험':<9} {'dropout':>8} {'tying':>6} {'schedule':>9} "
          f"{'학습PPL':>9} {'검증PPL':>9} {'격차':>7} {'파라미터':>10}")
    print("  " + "-" * 76)
    for r in results:
        gap = r["valid_ppl"] / r["train_ppl"]
        print(f"  {r['tag']:<9} {r['dropout']:>8} {str(r['tying']):>6} {r['schedule']:>9} "
              f"{r['train_ppl']:>9.4f} {r['valid_ppl']:>9.4f} {gap:>6.2f}배 {r['params']:>10,}")

    base = next((r for r in results if r["tag"] == "none"), None)
    if base:
        print(f"\n  기준선(none) 대비 검증 PPL 변화:")
        for r in results:
            if r["tag"] == "none":
                continue
            d = r["valid_ppl"] - base["valid_ppl"]
            mark = "좋아짐" if d < 0 else "나빠짐"
            print(f"    {r['tag']:<9} {d:+.4f}  ({mark})")
    print("\n  ※ tying 은 파라미터를 V×H 만큼 줄여요 — PPL 과 크기를 **함께** 보세요.")


def main(argv):
    wanted = None
    if "--runs" in argv:
        wanted = {a for a in argv[argv.index("--runs") + 1:] if not a.startswith("-")}
    json_path = argv[argv.index("--json") + 1] if "--json" in argv else None

    results = []
    for tag, tools, note in RUNS:
        if wanted and tag not in wanted:
            continue
        results.append(run_one(tag, tools, note))
        if json_path:                       # 도중에 끊겨도 여기까지는 남도록
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)

    if results:
        report(results)
    if json_path:
        print(f"\n결과 저장 -> {json_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
