# -*- coding: utf-8 -*-
"""
lm.py  (v0.2.5)  -  3자 비교 (캡스톤): 카운트 vs one-hot 신경망 vs 임베딩 MLP

[이 버전의 '새 개념' 은 모델이 아니라 '질문' 이에요]
  모델·학습·저장은 v0.2.4 를 **그대로** 물려받아요 (임베딩 MLP + 튜닝 3종).
  v0.2.5 가 더하는 건 — 임베딩 시대를 열면서 내걸었던 약속을 **실제로 재보는 것**:

      "one-hot 은 처음 보는 조합에 무력하지만,
       임베딩은 비슷한 토큰이 비슷한 벡터라 **처음 보는 조합에도 일반화**한다."   (v0.2.0)

  v0.1.5 의 대결표는 이 질문에 답할 수 없어요. 검증 PPL 하나는 **본 문맥과 처음 보는 문맥을
  섞어서 평균**낸 값이라, 어느 쪽에서 이겼는지가 안 보이거든요.

[그래서 자리를 나눠서 잽니다 — 이게 이 버전의 전부]
  검증 문장의 채점 자리를 **학습 데이터에서 봤는지**로 세 칸에 나눈 뒤, 칸마다 따로 PPL 을 재요.

    ① 조합까지 본 자리   : 그 문맥에서 그 다음 토큰이 학습에 **있었음**   → 외우기만 해도 맞힘
    ② 문맥은 봤으나 처음 : 문맥은 봤지만 그 다음 토큰 조합은 **처음**     → ★ 일반화가 필요한 자리
    ③ 문맥부터 처음      : 그 문맥 자체가 학습에 **없었음**               → ★★ 가장 어려운 자리

  ②③ 이 바로 '처음 보는 조합' 이에요. 임베딩의 주장이 맞다면 여기서 벌어져야 합니다.

[공정한 비교인 이유]
  · 세 모델 모두 같은 토크나이저(punct)·같은 prepare(<END>)·같은 FLOOR·같은 채점 자리를 씁니다.
  · '봤다/처음이다' 판정은 **모델과 무관**하게 데이터만 보고 정해요 (아래 NOVELTY_CONTEXT).
    어느 모델의 문맥 길이(1·2·3토큰)로도 편들지 않으려고 셋의 공통 분모인 **앞 2토큰**을 씁니다.
  · 자리 순서가 모델마다 어긋나면 비교가 깨지므로, 개수가 다르면 그 버전을 **건너뜁니다**.

[사용]  2.test/test.py 가 compare() 로 3자 대결표를 출력해요.

[주의]  신경망 버전을 부르려면 PyTorch 가 필요해요 (v0.0.9 카운트 모델은 표준 라이브러리만).
"""

import os
import math
import importlib.util

try:
    import torch
except ImportError:
    torch = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERSION_DIR = os.path.dirname(_HERE)


def _load_prev_module(prev_version):
    """이전 버전 lm.py 모듈을 통째로 불러와요 (NGramLM 계보 재사용)."""
    group = prev_version.rsplit(".", 1)[0]                    # "v0.2.4" -> "v0.2"
    llm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../llm 루트
    path = os.path.join(llm_dir, group, prev_version, "0.model", "lm.py")
    spec = importlib.util.spec_from_file_location("llmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# v0.2.4 를 그대로 물려받아요 (임베딩 MLP·튜닝·시각화 전부 동일).
_prev = _load_prev_module("v0.2.4")


class NeuralLM(_prev.NGramLM):
    pass          # 모델은 v0.2.4 그대로. 이 버전의 새 개념은 아래 '비교' 도구예요.


# web_service / 상속 사슬이 module.NGramLM 을 찾으므로 노출.
NGramLM = NeuralLM


class Model(NeuralLM):
    pass


# ----------------------------------------------------------------------
# 캡스톤 ① — 다른 버전 모델 불러오기
# ----------------------------------------------------------------------
def load_version_model(version):
    """
    다른 버전의 학습 결과를 **그 버전 클래스로** 불러와요.
      load_version_model("v0.0.9")  -> 카운트 (model.json)
      load_version_model("v0.1.5")  -> one-hot 신경망 (model.pt + vocab.json)
      load_version_model("v0.2.5")  -> 임베딩 MLP (model.pt + vocab.json)
    """
    group = version.rsplit(".", 1)[0]
    llm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))
    vdir = os.path.join(llm_dir, group, version)
    spec = importlib.util.spec_from_file_location(
        "llm_cmp_" + version.replace(".", "_"), os.path.join(vdir, "0.model", "lm.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mdir = os.path.join(vdir, "0.model")
    pt = os.path.join(mdir, "model.pt")
    model_file = pt if os.path.exists(pt) else os.path.join(mdir, "model.json")
    if not os.path.exists(model_file):
        raise FileNotFoundError(model_file)
    return module.NGramLM().load(model_file)


def param_count(lm):
    """파라미터 수. 신경망은 텐서 원소 수, 카운트는 **표의 칸 수**(같은 뜻의 '외운 양')."""
    if getattr(lm, "net", None) is not None:
        return sum(p.numel() for p in lm.net.parameters())
    return sum(len(counts) for table in lm._tables().values() for counts in table.values())


# ----------------------------------------------------------------------
# 캡스톤 ② — 채점 자리마다 log p 를 뽑기 (모델 종류에 관계없이 같은 순서로)
# ----------------------------------------------------------------------
# '봤다/처음이다' 를 판정할 때 볼 문맥 길이.
# 카운트(최대 2토큰) · one-hot 신경망(2토큰) · 임베딩 MLP(3토큰) 의 **공통 분모**라
# 어느 한쪽에 유리하지 않아요.
NOVELTY_CONTEXT = 2


def scored_positions(lm, sentences):
    """
    perplexity() 가 점수를 매기는 자리를 **그 순서 그대로** 내놓아요 — (tokens, i).
    (문맥이 없는 맨 앞 토큰은 제외 = v0.0.9 이후 모든 버전의 공통 규칙)
    """
    for sentence in sentences:
        tokens = lm.prepare(lm.tokenize(sentence))
        for i in range(1, len(tokens)):
            yield tokens, i


def token_logprobs(lm, sentences):
    """
    채점 자리마다 log p(정답) 를 하나씩, `scored_positions` 와 같은 순서로 돌려줘요.
    exp(-평균) 하면 그 버전의 perplexity() 와 **정확히 같은 값**이 나옵니다.

    신경망이면 perplexity() 와 똑같이 **한 번에 배치로** 통과시켜요.
    (자리마다 신경망을 따로 부르면 GPU 왕복 때문에 수십 배 느려요 — v0.1.0 에서 배운 그 이유.)
    """
    floor_log = math.log(lm.FLOOR)

    # --- 카운트 모델: 표 찾기라 자리별로 불러도 빨라요 ---
    if getattr(lm, "net", None) is None:
        return [math.log(lm.token_prob(tokens[:i], tokens[i]))
                for tokens, i in scored_positions(lm, sentences)]

    # --- 신경망: perplexity() 와 같은 방식으로 배치 처리 ---
    if torch is None:
        raise SystemExit("이 비교에는 PyTorch 가 필요해요.\n  pip install torch")

    logps, rows, targets, slots = [], [], [], []
    for tokens, i in scored_positions(lm, sentences):
        ids = lm._context_ids(tokens[:i])
        if tokens[i] not in lm.stoi or ids is None:
            logps.append(floor_log)              # token_prob 가 FLOOR 를 주는 자리와 동일
        else:
            slots.append(len(logps))
            logps.append(None)                   # 배치로 계산해 아래에서 채워요
            rows.append(ids)
            targets.append(lm.stoi[tokens[i]])

    if rows:
        x = torch.tensor(rows, dtype=torch.long, device=lm.device())
        y = torch.tensor(targets, dtype=torch.long, device=lm.device())
        was_training = lm.net.training
        lm.net.eval()
        with torch.no_grad():
            for k in range(0, len(targets), lm.EVAL_BATCH):
                logits = lm.net(x[k:k + lm.EVAL_BATCH])
                probs = torch.softmax(logits, dim=1)
                p = probs.gather(1, y[k:k + lm.EVAL_BATCH, None]).squeeze(1)
                # 카운트와 같은 바닥값으로 눌러 log(0) 을 막아요 (token_prob 와 동일).
                # 합산은 CPU 배정밀도 — MPS 는 float64 가 없고 float32 는 오차가 쌓여요.
                p = p.cpu().double().clamp(min=lm.FLOOR)
                for j, lp in enumerate(torch.log(p).tolist()):
                    logps[slots[k + j]] = lp
        if was_training:
            lm.net.train()

    return logps


# ----------------------------------------------------------------------
# 캡스톤 ③ — 자리를 '얼마나 처음 보는가' 로 나누기
# ----------------------------------------------------------------------
SEEN_PAIR = "본 조합"          # ① 문맥+다음토큰 둘 다 학습에 있었음
NEW_PAIR = "새 조합"           # ② 문맥은 봤지만 그 조합은 처음
NEW_CONTEXT = "새 문맥"        # ③ 문맥 자체가 처음
BUCKETS = (SEEN_PAIR, NEW_PAIR, NEW_CONTEXT)


def novelty_buckets(reader, train_sentences, valid_sentences, order=NOVELTY_CONTEXT):
    """
    검증 채점 자리마다 어느 칸인지 이름표를 붙여요 — `scored_positions` 와 같은 순서.

    판정은 **데이터만** 보고 합니다(모델을 전혀 안 봐요). 학습 문장에서 나온
    (앞 order토큰, 다음토큰) 을 모아 두고, 검증 자리를 그 집합과 대조할 뿐이에요.
    """
    seen_ctx, seen_pair = set(), set()
    for tokens, i in scored_positions(reader, train_sentences):
        ctx = tuple(tokens[max(0, i - order):i])
        seen_ctx.add(ctx)
        seen_pair.add((ctx, tokens[i]))

    labels = []
    for tokens, i in scored_positions(reader, valid_sentences):
        ctx = tuple(tokens[max(0, i - order):i])
        if (ctx, tokens[i]) in seen_pair:
            labels.append(SEEN_PAIR)
        elif ctx in seen_ctx:
            labels.append(NEW_PAIR)
        else:
            labels.append(NEW_CONTEXT)
    return labels


def bucket_ppl(logps, labels):
    """칸마다 PPL = exp(-평균 log p). 돌려주는 것: {칸이름: (ppl, 자리 수)}."""
    sums, counts = {b: 0.0 for b in BUCKETS}, {b: 0 for b in BUCKETS}
    for lp, b in zip(logps, labels):
        sums[b] += lp
        counts[b] += 1
    return {b: ((math.exp(-sums[b] / counts[b]) if counts[b] else float("nan")), counts[b])
            for b in BUCKETS}


# ----------------------------------------------------------------------
# 캡스톤 ④ — 3자 대결
# ----------------------------------------------------------------------
def compare(reader, train_sentences, valid_sentences, versions):
    """
    각 버전을 같은 검증 자리에서 재고, **칸별로 나눈** 결과까지 돌려줘요.

    돌려주는 것: (rows, skipped)
      rows = [{"v", "kind", "ppl", "buckets", "params", "top1", "coverage"}, ...]
    """
    labels = novelty_buckets(reader, train_sentences, valid_sentences)
    rows, skipped = [], []
    for version in versions:
        try:
            lm = load_version_model(version)
        except SystemExit:
            skipped.append((version, "PyTorch 필요 (pip install torch)"))
            continue
        except FileNotFoundError:
            skipped.append((version, "아직 학습 안 됨 — 그 버전 1.train/train.py 실행"))
            continue

        logps = token_logprobs(lm, valid_sentences)
        if len(logps) != len(labels):
            # 토크나이저나 prepare 가 다르면 자리가 어긋나요 = 사과 대 오렌지
            skipped.append((version, f"채점 자리 수가 달라요 ({len(logps)} vs {len(labels)})"))
            continue

        acc = lm.accuracy(valid_sentences)
        rows.append({"v": version,
                     "kind": _kind(version),
                     "ppl": math.exp(-sum(logps) / len(logps)),
                     "buckets": bucket_ppl(logps, labels),
                     "params": param_count(lm),
                     "top1": acc["top1"],
                     "coverage": acc["coverage"]})
    return rows, skipped


def _kind(version):
    if version.startswith("v0.0"):
        return "카운트"
    if version.startswith("v0.1"):
        return "one-hot"
    return "임베딩"


_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_VERSION_DIR)))  # 저장소 루트
_DATA_DIR = os.path.join(_ROOT, "data")                                  # 모든 버전 공용 데이터
DATA_PATH = os.path.join(_DATA_DIR, "pretrain", "train.txt")      # 학습용
VALID_PATH = os.path.join(_DATA_DIR, "pretrain", "valid.txt")    # 검증용
MODEL_PATH = os.path.join(_HERE, "model.pt")         # 가중치 (PyTorch 표준)
VOCAB_PATH = os.path.join(_HERE, "vocab.json")       # 어휘 (0번=<PAD>)
