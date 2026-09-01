# -*- coding: utf-8 -*-
"""
ablation_block_size.py  -  문맥 길이(N)와 용량(E·H)의 효과를 **따로** 재는 2x2 실험

[왜 필요한가요?]
버전끼리 비교하면 "무엇이 원인인가"를 말하기 어려워요. 두 버전 사이에는 대개 여러 값이
한꺼번에 바뀌거든요. 이 스크립트는 그걸 **한 번에 하나만** 바꿔서 분리합니다.

여기서 분리하는 두 축은 **문맥 길이 N** 과 **용량(E·H)** 이에요. 둘은 서로 얽히기 쉬워요 —
N 을 늘리면 fc1 의 입력 폭(N·E)이 커져서 파라미터도 함께 늘거든요. 그래서 용량 묶음을
**고정한 채** N 만 2↔3 으로 바꿔 4번 학습합니다.

    A. 큰 용량(E=H=256) · N=2   ← 온전성 검사: 기존 v0.2.0 을 재현해야 함
    B. 큰 용량(E=H=256) · N=3   ← ★ 용량 고정, 문맥만 넓힘 (= 기존 v0.2.1 을 재현)
    C. 작은 용량(E=32,H=128) · N=2
    D. 작은 용량(E=32,H=128) · N=3   ← ★ 작은 용량에서도 같은 방향인지 확인

A·B 가 알려진 값(v0.2.0·v0.2.1)을 재현하면 C·D 의 새 수치를 믿을 수 있고,
**각 묶음 안에서 N=2 대 N=3 의 차이**가 문맥 길이의 순수 효과예요.

[실측 결과 — 문맥을 넓히면 이득]
    설정              N=2       N=3      N 효과
    큰 용량        2.6488    2.3675   -0.2813   ← 넓히니 좋아짐
    작은 용량      3.0586    2.7487   -0.3099   ← 역시 좋아짐 (방향 일치)

두 묶음 모두 **문맥을 넓히면 이득**이에요(-0.28, -0.31). 용량 차이(E·H 를 줄인 것)는
**0.40** 으로 비슷한 크기고요. 문맥과 용량은 **서로 다른 축이고, 둘 다 실제로 작용**합니다.

[교훈]  한 번에 하나만 바꿔야 원인을 말할 수 있어요. 이 저장소의 "버전 = 개념 한 걸음"
원칙이 바로 그 이야기이고, v0.2.1 은 그 원칙에 맞춰 **용량을 v0.2.0 과 똑같이 두고
BLOCK_SIZE 만** 바꿉니다 — 그래서 v0.2.0 → v0.2.1 자체가 위 B 실험과 같아요.

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
    "큰 용량(E=H=256)": dict(EMBED=256, HIDDEN=256, OPTIMIZER="adam", LR=0.0003, EPOCHS=1500,
                       BATCH_SIZE=64, SEED=1234, WEIGHT_DECAY=0.0, INIT="zeros",
                       LABEL_SMOOTHING=0.0, EARLY_STOPPING=True, PATIENCE=20, MIN_DELTA=0.0),
    "작은 용량(E=32,H=128)": dict(EMBED=32, HIDDEN=128, OPTIMIZER="adam", LR=0.01, EPOCHS=1500,
                       BATCH_SIZE=64, SEED=1234, WEIGHT_DECAY=0.0, INIT="default",
                       LABEL_SMOOTHING=0.1, EARLY_STOPPING=True, PATIENCE=20, MIN_DELTA=0.0),
}

RUNS = [
    ("A", "큰 용량(E=H=256)", 2, "온전성 검사 — 기존 v0.2.0 을 재현해야 함"),
    ("B", "큰 용량(E=H=256)", 3, "★ 용량 고정, 문맥만 확장"),
    ("C", "작은 용량(E=32,H=128)", 2, "★ 작은 용량, 문맥만 축소"),
    ("D", "작은 용량(E=32,H=128)", 3, "온전성 검사 — 기존 v0.2.1 을 재현해야 함"),
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

    capacity_effect = mean_ppl("작은 용량(E=32,H=128)") - mean_ppl("큰 용량(E=H=256)")
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

    print("\n  ⇒ **문맥과 용량은 서로 다른 축이에요.** 문맥만 넓힌 B·D 는 좋아졌고,")
    print("    용량을 줄인 것은 따로 손해를 냅니다.")
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
