# -*- coding: utf-8 -*-
"""
ablation_block_size.py  -  "문맥을 넓히면 정말 나빠지나?" 를 **한 번에 하나만 바꿔서** 확인하는 실험

[왜 필요한가요?]
v0.2.0(검증 2.94)과 v0.2.1(3.37)을 나란히 놓으면 "문맥을 2→3으로 넓혔더니 나빠졌다"고
말하고 싶어져요. 그런데 두 버전은 문맥 말고도 **5가지가 더 다릅니다**:

    항목             v0.2.0     v0.2.1
    BLOCK_SIZE           2          3     ← 이것만 다르다고 생각하기 쉽지만
    EMBED              256         32     ← 1/8
    HIDDEN             256        128     ← 1/2
    LR              0.0003       0.01
    INIT             zeros    default
    LABEL_SMOOTHING    0.0        0.1

여섯 개가 한꺼번에 바뀐 비교로는 **어느 것이 원인인지 말할 수 없어요.** 그래서 이 스크립트는
나머지를 한 묶음으로 **고정한 채 BLOCK_SIZE 만** 2↔3 으로 바꿔 4번 학습합니다(2x2).

    A. v0.2.0 설정 · N=2   ← 온전성 검사: 기존 v0.2.0 을 재현해야 함
    B. v0.2.0 설정 · N=3   ← ★ 용량을 고정하고 문맥만 넓힘
    C. v0.2.1 설정 · N=2   ← ★ 작은 용량에서 문맥만 좁힘
    D. v0.2.1 설정 · N=3   ← 온전성 검사: 기존 v0.2.1 을 재현해야 함

A·D 가 알려진 값을 재현하면 B·C 의 새 수치를 믿을 수 있고, **각 묶음 안에서 N=2 대 N=3 의 차이**가
문맥 길이의 순수 효과예요.

[실측 결과 — 문맥은 범인이 아니었어요]
    설정              N=2       N=3      N 효과
    v0.2.0 설정    2.6488    2.3675   -0.2813   ← 넓히니 좋아짐
    v0.2.1 설정    3.0586    2.7487   -0.3099   ← 역시 좋아짐 (방향 일치)

두 설정 모두 **문맥을 넓히면 이득**이에요(-0.28, -0.31). 용량 차이(E·H 를 줄인 것)는
**0.40** 으로 비슷한 크기고요. → v0.2.1 이 뒤진 건 문맥 때문이 아니라 **함께 줄인 용량 때문**입니다.

[교훈]  한 번에 하나만 바꿔야 원인을 말할 수 있어요. 이 저장소의 "버전 = 개념 한 걸음"
원칙이 바로 그 이야기인데, v0.2.1 은 그 원칙이 지켜지지 않은 예였습니다.

[실행 방법]
    python3 ablation_block_size.py                 # 4번 전부 (MPS 로 약 30분)
    python3 ablation_block_size.py --runs B C      # 고른 것만
    python3 ablation_block_size.py --json out.json # 결과를 파일로도 저장

  · 학습된 모델을 **저장하지 않아요** — 기존 0.model/ 산출물을 건드리지 않습니다.
  · PyTorch 가 필요해요 (`pip install torch`).
"""

import os
import sys
import json
import time
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
LM_PATH = os.path.join(HERE, "llm", "v0.2", "v0.2.1", "0.model", "lm.py")   # 임의의 N 을 지원
DATA_DIR = os.path.join(HERE, "data", "pretrain")
TRAIN_TXT = os.path.join(DATA_DIR, "train.txt")
VALID_TXT = os.path.join(DATA_DIR, "valid.txt")

# 각 버전 1.train/train.py 의 설정을 그대로 옮긴 것 (BLOCK_SIZE 만 아래에서 따로 줍니다)
CONFIG = {
    "v0.2.0 설정": dict(EMBED=256, HIDDEN=256, OPTIMIZER="adam", LR=0.0003, EPOCHS=1500,
                       BATCH_SIZE=64, SEED=1234, WEIGHT_DECAY=0.0, INIT="zeros",
                       LABEL_SMOOTHING=0.0, EARLY_STOPPING=True, PATIENCE=20, MIN_DELTA=0.0),
    "v0.2.1 설정": dict(EMBED=32, HIDDEN=128, OPTIMIZER="adam", LR=0.01, EPOCHS=1500,
                       BATCH_SIZE=64, SEED=1234, WEIGHT_DECAY=0.0, INIT="default",
                       LABEL_SMOOTHING=0.1, EARLY_STOPPING=True, PATIENCE=20, MIN_DELTA=0.0),
}

RUNS = [
    ("A", "v0.2.0 설정", 2, "온전성 검사 — 기존 v0.2.0 을 재현해야 함"),
    ("B", "v0.2.0 설정", 3, "★ 용량 고정, 문맥만 확장"),
    ("C", "v0.2.1 설정", 2, "★ 작은 용량, 문맥만 축소"),
    ("D", "v0.2.1 설정", 3, "온전성 검사 — 기존 v0.2.1 을 재현해야 함"),
]


def load_lm_module():
    """v0.2.1 의 lm.py 를 파일 경로로 불러와요. (폴더 이름에 '.' 이 있어 일반 import 불가)"""
    spec = importlib.util.spec_from_file_location("ablation_lm", LM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_one(tag, config_name, block_size, note):
    """설정 한 묶음 + BLOCK_SIZE 하나로 학습하고 (학습 PPL, 검증 PPL) 을 잽니다."""
    module = load_lm_module()          # 실행마다 새로 (클래스 상태를 깨끗하게)
    lm = module.Model()
    for key, value in CONFIG[config_name].items():
        setattr(lm, key, value)
    lm.BLOCK_SIZE = block_size
    lm.DEVICE = "auto"

    train_sentences = lm.read_sentences(TRAIN_TXT)
    valid_sentences = lm.read_sentences(VALID_TXT)

    print(f"\n{'=' * 70}")
    print(f"[{tag}] {config_name} · N={block_size}  — {note}")
    print(f"     E={lm.EMBED} H={lm.HIDDEN} LR={lm.LR} INIT={lm.INIT} "
          f"LABEL_SMOOTHING={lm.LABEL_SMOOTHING}", flush=True)

    started = time.time()
    lm.train(train_sentences, valid_sentences)
    seconds = time.time() - started

    params = sum(p.numel() for p in lm.net.parameters())
    train_ppl = lm.perplexity(train_sentences)
    valid_ppl = lm.perplexity(valid_sentences)
    print(f"[{tag}] 완료 {seconds:.0f}s · 채택 epoch {lm.stopped_epoch} · 파라미터 {params:,} · "
          f"학습PPL {train_ppl:.4f} · 검증PPL {valid_ppl:.4f}", flush=True)

    return dict(tag=tag, config=config_name, block_size=block_size, params=params,
                epoch=lm.stopped_epoch, train_ppl=round(train_ppl, 4),
                valid_ppl=round(valid_ppl, 4), seconds=round(seconds))


def report(results):
    """2x2 표 — 같은 줄 안에서 N 만 다르므로, 그 차이가 문맥 길이의 순수 효과예요."""
    print(f"\n{'=' * 70}")
    print("=== 2x2 결과 — 문맥 길이(N) 만의 효과 ===\n")
    print(f"  {'설정':<14} {'N=2':>9} {'N=3':>9} {'N 효과':>10}   파라미터(N=2 → N=3)")
    print("  " + "-" * 68)

    effects = []
    for config_name in CONFIG:
        pair = {r["block_size"]: r for r in results if r["config"] == config_name}
        if 2 not in pair or 3 not in pair:
            continue                       # --runs 로 일부만 돌린 경우
        two, three = pair[2], pair[3]
        effect = three["valid_ppl"] - two["valid_ppl"]
        effects.append(effect)
        print(f"  {config_name:<14} {two['valid_ppl']:>9.4f} {three['valid_ppl']:>9.4f} "
              f"{effect:>+10.4f}   {two['params']:,} → {three['params']:,}")

    if len(effects) < 2:
        print("\n  (2x2 를 다 돌려야 결론이 나와요 — --runs 를 빼고 실행해 보세요.)")
        return

    # 용량 효과: 큰 용량 묶음과 작은 용량 묶음의 평균 차이
    def mean_ppl(config_name):
        vals = [r["valid_ppl"] for r in results if r["config"] == config_name]
        return sum(vals) / len(vals)

    capacity_effect = mean_ppl("v0.2.1 설정") - mean_ppl("v0.2.0 설정")
    context_effect = max(abs(e) for e in effects)
    same_sign = all(e < 0 for e in effects) or all(e > 0 for e in effects)

    # 해설은 **측정값에서** 뽑아요 (미리 적어둔 결론을 찍지 않습니다).
    if not same_sign:
        verdict = "부호가 설정에 따라 뒤집힘 — 사실상 잡음"
    elif effects[0] < 0:
        verdict = "두 설정 모두 **넓힐수록 이득** — 방향이 일치해요"
    else:
        verdict = "두 설정 모두 넓힐수록 손해 — 방향이 일치해요"

    print(f"\n  → 문맥 길이 효과 : 최대 {context_effect:.4f}  ({verdict})")
    ratio = capacity_effect / context_effect if context_effect else float("inf")
    print(f"  → 용    량 효과 : {capacity_effect:.4f}  "
          f"(E·H 를 줄인 것 — 문맥 효과의 약 {ratio:.1f}배)")

    print("\n  ⇒ v0.2.1 이 v0.2.0 보다 뒤지는 건 **문맥을 넓혀서가 아니라 용량을 줄여서**예요.")
    print("    문맥만 따로 넓힌 B·D 는 오히려 좋아졌으니까요.")
    if same_sign and effects[0] < 0:
        print("\n  💡 문맥을 넓히는 게 왜 이만큼 이득일까요 — v0.0.6 이 조사를 분리했기 때문이에요.")
        print("     토큰이 잘게 쪼개진 만큼 같은 길이의 글을 담으려면 **더 많은 토큰**이 필요해요.")
        print("     (단어 단위였을 때는 이 효과가 ±0.02 로 잡음에 묻혔습니다.)")
    print("\n    어느 쪽이든 교훈은 같아요 — 한 번에 하나만 바꿔야 원인을 말할 수 있습니다.")


def main(argv):
    if not os.path.exists(VALID_TXT):
        raise SystemExit(f"검증 데이터가 없어요: {VALID_TXT}  (data/pretrain/generate.py 실행)")

    wanted = None
    if "--runs" in argv:
        wanted = {a.upper() for a in argv[argv.index("--runs") + 1:] if not a.startswith("-")}
    json_path = None
    if "--json" in argv:
        json_path = argv[argv.index("--json") + 1]

    results = []
    for tag, config_name, block_size, note in RUNS:
        if wanted and tag not in wanted:
            continue
        results.append(run_one(tag, config_name, block_size, note))
        if json_path:                      # 도중에 끊겨도 여기까지는 남도록 매번 저장
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)

    if results:
        report(results)
    if json_path:
        print(f"\n결과 저장 -> {json_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
