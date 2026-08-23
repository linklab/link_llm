# -*- coding: utf-8 -*-
"""
eval_suite.py  -  여러 버전을 한 번에, **여러 지표로** 비교하는 평가 도구

[왜 필요한가요?]
각 버전의 `4.test/test.py` 는 '그 버전 하나'를 봅니다. 이 도구는 학습된 버전을 **모두 모아**
같은 데이터로 나란히 재요. 그리고 무엇보다 **지표를 하나만 보지 않습니다.**

  ① PPL          : 확률을 얼마나 잘 배분했나        (낮을수록 좋음)
  ② 정확도        : 1등을 얼마나 맞혔나              (높을수록 좋음)
  ③ 후보 포함률   : 정답을 만들어낼 수 있기라도 한가  (카운트의 백오프 한계가 보임)
  ④ 답변 고르기   : 진짜 답을 가짜 답들 사이에서 고르나 (대화 모델다운 평가)
  ⑤ 생성 다양성   : 온도를 바꾸면 답이 실제로 다양해지나 (PPL 이 절대 못 보는 것)
  ⑥ 크기          : 파라미터 수 · 파일 크기            (공짜 성능은 없어요)

①②③ 은 각 버전 `2.models/lm.py` 가 이미 가진 `perplexity()` / `accuracy()` 를 그대로 부르고,
④⑤⑥ 은 여러 모델을 나란히 놓아야 의미가 생기는 것이라 이 파일에 있어요.

[핵심 - 지표마다 승자가 다릅니다]
  · PPL 1등은 v0.1.5(one-hot), top-1 정확도 1등은 v0.2.0(임베딩) 입니다.
  · 온도 0 에서 v0.1.x 1토큰 모델은 **모든 질문에 같은 답**을 합니다 — PPL 로는 안 보여요.
"이 모델이 더 좋다"고 말하려면 **어느 자로 쟀는지**를 같이 말해야 한다는 게 이 도구의 교훈이에요.

[실행 방법]
    python3 eval_suite.py                    # 학습된 버전 전부
    python3 eval_suite.py v0.0.9 v0.2.0      # 고른 버전만
    python3 eval_suite.py --quick            # 생성 지표(⑤)는 느리니 건너뛰기

  · 먼저 각 버전의 `3.train/train.py` 를 돌려 학습해 두어야 목록에 나와요.
  · 신경망(v0.1.x~) 버전을 포함하려면 PyTorch 가 필요해요 (`pip install torch`).
    없으면 그 버전만 자동으로 건너뜁니다.

[공정한 비교를 위해]
  · 모든 버전을 **같은 검증 데이터**로 잽니다 (기준 버전의 `1.data/valid.txt`).
  · 학습 데이터가 기준과 다른 옛 버전(v0.0.1~v0.0.7)은 사과 대 오렌지라 자동 제외해요.
"""

import os
import sys
import math
import random
import hashlib
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
LLM_DIR = os.path.join(HERE, "llm")

REFERENCE = "v0.2.0"      # 데이터·비교의 기준이 되는 버전 (가장 최신)
N_CANDIDATES = 10         # ④ 답변 고르기: 진짜 1개 + 가짜 9개
TEMPERATURES = (0.0, 0.7, 1.0)   # ⑤ 생성 다양성을 재볼 온도들
SEED = 1234


# ----------------------------------------------------------------------
# 버전 찾기 / 불러오기
#   (폴더 이름에 '.' 이 있어 보통의 import 가 안 되므로 파일 경로로 불러와요.
#    각 버전 lm.py 가 쓰는 방식과 같습니다.)
# ----------------------------------------------------------------------
def version_dir(version):
    return os.path.join(LLM_DIR, version.rsplit(".", 1)[0], version)


def version_key(version):
    try:
        return tuple(int(p) for p in version.lstrip("v").split("."))
    except ValueError:
        return ()


def model_path(version):
    """학습 결과 파일. 신경망은 model.pt, 개수 세기는 model.json."""
    mdir = os.path.join(version_dir(version), "2.models")
    pt = os.path.join(mdir, "model.pt")
    return pt if os.path.exists(pt) else os.path.join(mdir, "model.json")


def all_versions():
    """llm/ 아래에서 **학습이 끝난** 버전을 모두 찾아 번호순으로."""
    found = []
    for group in sorted(os.listdir(LLM_DIR)):
        gdir = os.path.join(LLM_DIR, group)
        if not (group.startswith("v") and os.path.isdir(gdir)):
            continue
        for name in os.listdir(gdir):
            if name.startswith("v") and os.path.exists(model_path(name)):
                found.append(name)
    return sorted(found, key=version_key)


def data_fingerprint(version):
    """그 버전 학습 데이터의 지문(해시). 기준과 다르면 비교에서 빼요."""
    path = os.path.join(version_dir(version), "1.data", "data.txt")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_model(version):
    """그 버전의 lm.py 를 불러와 학습된 모델을 복원합니다."""
    path = os.path.join(version_dir(version), "2.models", "lm.py")
    spec = importlib.util.spec_from_file_location("eval_" + version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NGramLM().load(model_path(version))


# ----------------------------------------------------------------------
# ① PPL — 전체 / 'OOV 정답' 을 뺀 것
# ----------------------------------------------------------------------
def perplexities(lm, sentences, known):
    """
    (전체 PPL, OOV 정답을 뺀 PPL) 을 함께 돌려줍니다.

    검증 데이터에는 학습에서 한 번도 못 본 토큰(OOV)이 섞여 있어요. 그 자리는 어떤 모델이든
    똑같이 FLOOR 벌점을 먹기 때문에, 전체 PPL 은 모델 간 차이를 그만큼 희석시킵니다.
    'OOV 를 뺀 PPL' 은 **모델이 실제로 겨룰 수 있는 자리에서만** 비교한 점수예요.
    """
    tot_lp = tot_n = inv_lp = inv_n = 0
    for sentence in sentences:
        tokens = lm.prepare(lm.tokenize(sentence))
        for i in range(1, len(tokens)):
            lp = math.log(lm.token_prob(tokens[:i], tokens[i]))
            tot_lp += lp
            tot_n += 1
            if tokens[i] in known:
                inv_lp += lp
                inv_n += 1
    ppl_all = math.exp(-tot_lp / tot_n) if tot_n else float("inf")
    ppl_inv = math.exp(-inv_lp / inv_n) if inv_n else float("inf")
    return ppl_all, ppl_inv, tot_n - inv_n


def known_vocab(lm, train_sentences):
    """학습 데이터에 등장한 토큰 집합 (= 이 모델이 아는 단어)."""
    vocab = set()
    for sentence in train_sentences:
        vocab.update(lm.prepare(lm.tokenize(sentence)))
    return vocab


# ----------------------------------------------------------------------
# ④ 답변 고르기 (대화 모델다운 평가)
# ----------------------------------------------------------------------
def response_ranking(lm, valid_sentences, n_candidates=N_CANDIDATES, seed=SEED):
    """
    "<사용자> X <봇> Y" 에서 진짜 답 Y 1개 + 다른 대화의 가짜 답 (n-1)개를 섞어 놓고,
    모델이 **진짜를 1등으로 고르는지** 봅니다. (점수 = 길이로 나눈 평균 로그확률)

    생성이 아니라 **고르기**라서, 문장을 잘 못 만드는 모델도 공정하게 평가돼요.
    찍으면 1/n (10지선다면 10%) 이니, 그보다 높아야 의미가 있습니다.
    """
    rng = random.Random(seed)
    pairs = []
    for sentence in valid_sentences:
        if lm.BOT in sentence:
            user, bot = sentence.split(lm.BOT, 1)
            pairs.append((user.replace(lm.USER, "").strip(), bot.strip()))
    if not pairs:
        return None

    replies = [b for _, b in pairs]
    hit1 = hit3 = 0
    for user, gold in pairs:
        others = [r for r in replies if r != gold]
        candidates = [gold] + rng.sample(others, min(n_candidates - 1, len(others)))

        scored = []
        for candidate in candidates:
            context = lm.build_chat_context(None, user)      # [<사용자> ... <봇>]
            total, count = 0.0, 0
            for token in lm.tokenize(candidate) + [lm.END]:
                total += math.log(lm.token_prob(context, token))
                count += 1
                context = context + [token]
            scored.append((total / count, candidate))        # 길이 정규화 (긴 답이 불리하지 않게)

        scored.sort(key=lambda x: -x[0])
        if scored[0][1] == gold:
            hit1 += 1
        if gold in [c for _, c in scored[:3]]:
            hit3 += 1
    return {"r1": hit1 / len(pairs), "r3": hit3 / len(pairs), "n": len(pairs)}


# ----------------------------------------------------------------------
# ⑤ 생성 다양성 (PPL 이 절대 못 보는 것)
# ----------------------------------------------------------------------
def diversity(lm, prompts, temperature, seed=SEED):
    """
    질문들에 실제로 답을 만들어 보고, 그 답들이 얼마나 **다양한지** 잽니다.

      distinct-1 / distinct-2 : 쓰인 (단어 / 단어쌍) 중 서로 다른 것의 비율
      unique                  : 서로 다른 답의 비율

    온도 0(greedy)에서 이 값이 바닥이면 **모든 질문에 같은 답을 하는 붕괴** 상태예요.
    PPL 은 이걸 전혀 잡아내지 못합니다 — 그래서 따로 재요.
    """
    random.seed(seed)                       # 샘플링을 재현 가능하게
    outputs = []
    for prompt in prompts:
        try:
            outputs.append(lm.chat(prompt, history=None, temperature=temperature))
        except TypeError:                   # chat 시그니처가 다른 옛 버전 대비
            outputs.append(lm.chat(prompt, history=None))

    tokens, bigrams = [], []
    for out in outputs:
        toks = lm.tokenize(out)
        tokens += toks
        bigrams += list(zip(toks, toks[1:]))
    return {
        "d1": len(set(tokens)) / len(tokens) if tokens else 0.0,
        "d2": len(set(bigrams)) / len(bigrams) if bigrams else 0.0,
        "unique": len(set(outputs)) / len(outputs) if outputs else 0.0,
        "sample": outputs[0] if outputs else "",
    }


# ----------------------------------------------------------------------
# ⑥ 크기 — 공짜 성능은 없어요
# ----------------------------------------------------------------------
def model_size(lm, path):
    """(파일 크기 bytes, 파라미터 수). 신경망은 state_dict, 카운트는 표의 칸 수."""
    total_bytes = os.path.getsize(path)
    if os.path.basename(path) == "model.pt":
        vocab = os.path.join(os.path.dirname(path), "vocab.json")
        if os.path.exists(vocab):
            total_bytes += os.path.getsize(vocab)
        params = sum(p.numel() for p in lm.net.parameters())
    else:
        params = sum(len(counts) for table in lm._tables().values() for counts in table.values())
    return total_bytes, params


# ----------------------------------------------------------------------
# 실행
# ----------------------------------------------------------------------
def collect(versions, train_sentences, valid_sentences, prompts, quick):
    rows, skipped = [], []
    for version in versions:
        try:
            lm = load_model(version)
        except SystemExit:                  # 신경망인데 torch 가 없음
            skipped.append((version, "PyTorch 필요 (pip install torch)"))
            continue
        except FileNotFoundError:
            skipped.append((version, "아직 학습 안 됨 — 그 버전 3.train/train.py 실행"))
            continue
        if not hasattr(lm, "perplexity"):
            skipped.append((version, "평가 함수 없음 (퍼플렉서티는 v0.0.9 부터)"))
            continue

        known = known_vocab(lm, train_sentences)
        ppl_tr, _, _ = perplexities(lm, train_sentences, known)
        ppl_va, ppl_inv, n_oov = perplexities(lm, valid_sentences, known)
        acc = lm.accuracy(valid_sentences)
        rank = response_ranking(lm, valid_sentences) if hasattr(lm, "build_chat_context") else None
        size_bytes, params = model_size(lm, model_path(version))
        gen = None
        if not quick and hasattr(lm, "chat"):
            gen = {t: diversity(lm, prompts, t) for t in TEMPERATURES}

        rows.append({"v": version, "kind": "카운트" if version.startswith("v0.0") else "신경망",
                     "ppl_train": ppl_tr, "ppl_valid": ppl_va, "ppl_invocab": ppl_inv,
                     "oov": n_oov, "acc": acc, "rank": rank,
                     "bytes": size_bytes, "params": params, "gen": gen})
        print(f"  · {version} 완료", flush=True)
    return rows, skipped


def report(rows, skipped, train_sentences, valid_sentences, quick):
    if not rows:
        print("\n비교할 모델이 없어요. 먼저 각 버전의 3.train/train.py 를 실행해 주세요.")
        return

    n_scored = rows[0]["acc"]["n"]
    print(f"\n=== 버전 대결표 — 학습 {len(train_sentences)}문장 / 검증 {len(valid_sentences)}문장"
          f" (채점 {n_scored}자리) ===\n")

    head = (f"  {'버전':<8} {'종류':<5} {'학습PPL':>8} {'검증PPL':>8} {'OOV뺀':>8} "
            f"{'top-1':>7} {'top-5':>7} {'후보有':>7} {'답고르기':>8} {'파라미터':>10} {'크기':>8}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in rows:
        rank = f"{r['rank']['r1'] * 100:6.1f}%" if r["rank"] else "     - "
        print(f"  {r['v']:<8} {r['kind']:<5} {r['ppl_train']:8.2f} {r['ppl_valid']:8.2f} "
              f"{r['ppl_invocab']:8.2f} {r['acc']['top1'] * 100:6.1f}% {r['acc']['topk'] * 100:6.1f}% "
              f"{r['acc']['coverage'] * 100:6.1f}% {rank:>8} {r['params']:10,} "
              f"{r['bytes'] / 1024:7.0f}K")

    if skipped:
        print("\n  (제외됨)")
        for version, why in skipped:
            print(f"   - {version}: {why}")

    # ---- 지표별 승자 (동점이면 모두 적어요) ----
    def winners(score, best=max):
        top = best(score(r) for r in rows)
        names = [r["v"] for r in rows if abs(score(r) - top) < 1e-9]
        return top, ", ".join(names) if len(names) <= 3 else f"{names[0]} 외 {len(names) - 1}개"

    best_ppl = min(rows, key=lambda r: r["ppl_valid"])
    best_acc = max(rows, key=lambda r: r["acc"]["top1"])
    cov_top, cov_who = winners(lambda r: r["acc"]["coverage"])
    smallest = min(rows, key=lambda r: r["params"])
    print("\n=== 지표별 1등 ===")
    print(f"  검증 PPL 최저   : {best_ppl['v']} = {best_ppl['ppl_valid']:.2f}")
    print(f"  top-1 정확도    : {best_acc['v']} = {best_acc['acc']['top1'] * 100:.1f}%")
    print(f"  후보 포함률     : {cov_who} = {cov_top * 100:.1f}%")
    print(f"  가장 작은 모델  : {smallest['v']} = 파라미터 {smallest['params']:,}개")
    if best_ppl["v"] != best_acc["v"]:
        print(f"\n  → PPL 1등({best_ppl['v']})과 정확도 1등({best_acc['v']})이 **다릅니다**.")
        print("    '확률을 잘 배분하는 것'과 '1등을 잘 맞히는 것'은 다른 능력이에요.")
        print("    어떤 모델이 더 좋다고 말하려면 **어느 자로 쟀는지**를 같이 말해야 해요.")

    counts = [r for r in rows if r["kind"] == "카운트"]
    neurals = [r for r in rows if r["kind"] == "신경망"]
    if counts and neurals:
        c = max(counts, key=lambda r: r["acc"]["coverage"])
        n = max(neurals, key=lambda r: r["acc"]["coverage"])
        print(f"\n  → 후보 포함률: 카운트 {c['acc']['coverage'] * 100:.1f}% vs "
              f"신경망 {n['acc']['coverage'] * 100:.1f}%")
        print("    카운트는 표에 없는 조합을 아예 못 만들어요. PPL 에서는 FLOOR 가 메워주는 한계예요.")

    if rows[0]["rank"]:
        chance = 100.0 / N_CANDIDATES
        print(f"\n  → '답고르기'는 진짜 답 1개 + 가짜 {N_CANDIDATES - 1}개 중 고르기예요 "
              f"(찍으면 {chance:.0f}%).")
        print(f"    검증 대화가 {rows[0]['rank']['n']}개뿐이라 몇 %p 차이는 노이즈로 보세요.")

    # ---- 생성 다양성 ----
    if not quick and any(r["gen"] for r in rows):
        print("\n=== 생성 다양성 — PPL 이 못 보는 것 ===\n")
        print(f"  {'버전':<8} {'온도':>5} {'distinct-1':>11} {'distinct-2':>11} {'서로 다른 답':>12}   예시")
        print("  " + "-" * 78)
        for r in rows:
            if not r["gen"]:
                continue
            for t in TEMPERATURES:
                d = r["gen"][t]
                print(f"  {r['v']:<8} {t:>5.1f} {d['d1']:>11.3f} {d['d2']:>11.3f} "
                      f"{d['unique'] * 100:>11.0f}%   {d['sample'][:22]}")
        worst = min((r for r in rows if r["gen"]), key=lambda r: r["gen"][0.0]["unique"])
        print(f"\n  → 온도 0(greedy)에서 가장 심한 건 {worst['v']} — 질문이 달라도 "
              f"서로 다른 답이 {worst['gen'][0.0]['unique'] * 100:.0f}% 뿐이에요 "
              f"(사실상 모든 질문에 같은 답).")
        print("    온도를 올리면 다양해지지만 대신 엉뚱한 말이 늘어요 — 그 균형이 top-k/top-p 의 역할이에요.")


def main(argv):
    quick = "--quick" in argv
    asked = [a for a in argv if not a.startswith("-")]

    ref_dir = version_dir(REFERENCE)
    train_path = os.path.join(ref_dir, "1.data", "data.txt")
    valid_path = os.path.join(ref_dir, "1.data", "valid.txt")
    if not os.path.exists(valid_path):
        raise SystemExit(f"기준 버전({REFERENCE})의 검증 데이터가 없어요: {valid_path}")

    reference_data = data_fingerprint(REFERENCE)
    if asked:
        versions, skipped = asked, []
    else:
        versions, skipped = [], []
        for v in all_versions():
            if data_fingerprint(v) == reference_data:
                versions.append(v)
            else:
                skipped.append((v, "학습 데이터가 기준과 달라 공정 비교 불가"))

    if not versions:
        raise SystemExit("학습된 버전이 하나도 없어요. 먼저 3.train/train.py 를 실행해 주세요.")

    # 문장 읽기·토크나이즈는 어느 버전 클래스로 해도 같아서, 기준 버전 것을 씁니다.
    ruler = load_model(REFERENCE) if os.path.exists(model_path(REFERENCE)) else load_model(versions[-1])
    train_sentences = ruler.read_sentences(train_path)
    valid_sentences = ruler.read_sentences(valid_path)
    prompts = [s.split(ruler.BOT)[0].replace(ruler.USER, "").strip() for s in valid_sentences]

    print(f"기준 데이터: {REFERENCE}/1.data  ·  비교 대상 {len(versions)}개  "
          f"{'(--quick: 생성 지표 생략)' if quick else ''}")
    rows, more_skipped = collect(versions, train_sentences, valid_sentences, prompts, quick)
    report(rows, skipped + more_skipped, train_sentences, valid_sentences, quick)


if __name__ == "__main__":
    main(sys.argv[1:])
