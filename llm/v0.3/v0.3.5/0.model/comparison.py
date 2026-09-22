"""독립적인 후보 학습, 공통 채점 계약, 신규성별 평가와 재현 가능한 보고서."""
import argparse
import hashlib
import importlib.util
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


gpt = module("v035_gpt", HERE / "lm.py")
legacy_mlp = module("v035_mlp_arch", ROOT / "llm/v0.2/v0.2.1/0.model/lm.py")
count_module = module("v035_count", ROOT / "llm/v0.0/v0.0.9/0.model/lm.py")
BUCKETS = ("seen_pair", "new_pair", "new_context")
FLOOR = 1e-4


class SequenceMLP(nn.Module):
    """v0.2.1 concat→tanh MLP를 공통 (B,T,V) 학습 계약에 연결."""
    def __init__(self, vocab, embed, block_size, hidden):
        super().__init__()
        self.block_size = block_size
        self.core = legacy_mlp.BlockModel(vocab, hidden, embed, block_size)

    def forward(self, tokens):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.block_size:
            raise ValueError("입력은 (B,T), 1 <= T <= block_size")
        b, t = tokens.shape
        padded = torch.nn.functional.pad(tokens, (self.block_size - 1, 0))
        contexts = padded.unfold(1, self.block_size, 1)
        logits = self.core(contexts.reshape(b * t, self.block_size)).reshape(b, t, -1)
        return logits.masked_fill(torch.arange(logits.shape[-1], device=tokens.device).eq(0), -1e9)


class MLP(gpt.previous.NeuralLM):
    MODEL_VERSION = "v0.3.5-mlp-baseline"
    HIDDEN = 256

    def build_net(self):
        return SequenceMLP(len(self.itos), self.EMBED, self.BLOCK_SIZE, self.HIDDEN)

    def extra_metadata(self):
        return {"hidden": self.HIDDEN}

    def restore_extra_metadata(self, meta):
        self.HIDDEN = meta["hidden"]


class Count(count_module.Model):
    """가장 긴 가용 문맥의 정규화된 분포. 정답에 따른 백오프 없음."""
    def next_dist(self, recent):
        recent = recent[-self.BLOCK_SIZE:]
        if not recent or any(t not in self.known for t in recent):
            return None
        return super().next_dist(recent)

    def token_prob(self, recent, token):
        return max(FLOOR, (self.next_dist(recent) or {}).get(token, 0.0))


def positions(reader, sentences, context):
    return [(tuple(tokens[max(0, i-context):i]), tokens[i])
            for sentence in sentences
            for tokens in [reader.prepare(reader.tokenize(sentence))]
            for i in range(1, len(tokens))]


def novelty(reader, train, valid, context):
    pairs = set(positions(reader, train, context))
    contexts = {c for c, _ in pairs}
    return ["seen_pair" if p in pairs else "new_pair" if p[0] in contexts else "new_context"
            for p in positions(reader, valid, context)]


def assert_alignment(reader, lm, sentences):
    # 같은 개수라도 다른 토큰이면 비교를 거부합니다.
    for text in sentences:
        if reader.prepare(reader.tokenize(text)) != lm.prepare(lm.tokenize(text)):
            raise ValueError("토크나이저/END 채점 자리 불일치")


def scored(lm, sentences):
    """(log p, top1, top5, support) 순서. 동점은 어휘 순서로 고정."""
    output = []
    if isinstance(lm, Count):
        for ctx, gold in positions(lm, sentences, lm.BLOCK_SIZE):
            dist = lm.next_dist(ctx) or {}
            ranked = sorted(dist, key=lambda t: (-dist[t], lm.stoi[t]))[:5]
            output.append((math.log(max(FLOOR, dist.get(gold, 0))),
                           bool(ranked) and ranked[0] == gold, gold in ranked, gold in dist))
        return output
    was_training = lm.net.training
    lm.net.eval()
    try:
        with torch.no_grad():
            for x, y, known in lm.batches(lm.make_windows(sentences), lm.EVAL_BATCH):
                logits = lm.net(x.to(lm.device())).cpu()
                active = y.ne(-100)
                values, targets, good = logits[active], y[active], known[active]
                probs = values.softmax(-1).gather(1, targets[:, None]).squeeze(1).double()
                logs = probs.clamp_min(FLOOR).log()
                logs[~good] = math.log(FLOOR)
                ranked = values.argsort(dim=-1, descending=True, stable=True)[:, :min(5, len(lm.itos)-1)]
                for lp, rank, target, ok in zip(logs.tolist(), ranked.tolist(), targets.tolist(), good.tolist()):
                    output.append((lp, ok and rank[0] == target, ok and target in rank, ok))
    finally:
        lm.net.train(was_training)
    return output


def metrics(rows):
    n = len(rows)
    return {"n": n, "ppl": math.exp(-sum(r[0] for r in rows)/n) if n else None,
            "top1": sum(r[1] for r in rows)/n if n else None,
            "top5": sum(r[2] for r in rows)/n if n else None,
            "support": sum(r[3] for r in rows)/n if n else None}


def summarize(rows, labels):
    if len(rows) != len(labels):
        raise ValueError("채점 자리와 신규성 라벨 길이 불일치")
    if any(label not in BUCKETS for label in labels):
        raise ValueError("알 수 없는 신규성 라벨")
    return {"all": metrics(rows), "buckets": {
        label: metrics([r for r, b in zip(rows, labels) if b == label]) for label in BUCKETS}}


def generate(lm, prompt, temperature, seed, max_tokens=40):
    rng = random.Random(seed)
    tokens = lm.tokenize(prompt)
    continuation = []
    ended = False
    for _ in range(max_tokens):
        dist = lm.next_dist(tokens) or {}
        if not dist:
            break
        ranked = sorted(dist, key=lambda t: (-dist[t], lm.stoi[t]))
        if temperature == 0:
            token = ranked[0]
        else:
            logs = [math.log(max(dist[t], 1e-45))/temperature for t in ranked]
            peak = max(logs)
            token = rng.choices(ranked, weights=[math.exp(v-peak) for v in logs])[0]
        if lm.is_end(token):
            ended = True
            break
        tokens.append(token)
        continuation.append(token)
    return {"prompt": prompt, "temperature": temperature, "seed": seed,
            "continuation": lm.detokenize(continuation), "tokens": continuation,
            "ended": ended, "empty": not continuation}


def generation_report(lm, prompts, seed):
    result = {}
    for temp in (0.0, 0.7):
        samples = [generate(lm, p, temp, seed+i) for i, p in enumerate(prompts)]
        tokens = [t for s in samples for t in s["tokens"]]
        pairs = [p for s in samples for p in zip(s["tokens"], s["tokens"][1:])]
        result[str(temp)] = {"samples": samples, "distinct1": len(set(tokens))/len(tokens) if tokens else 0,
                            "distinct2": len(set(pairs))/len(pairs) if pairs else 0,
                            "empty_rate": sum(s["empty"] for s in samples)/len(samples)}
    return result


def candidates(context):
    # 후보 집합은 검증 결과를 보기 전에 고정합니다. 무제한 최대 성능은 아닙니다.
    return [(kind, size) for kind, sizes in (
        ("count", sorted({2, context})), ("mlp", sorted({2, context})),
        ("gpt", sorted({context, 32}))) for size in sizes]


def select(rows, context):
    fixed, best = [], []
    for seed in sorted({r["seed"] for r in rows}):
        for kind in ("count", "mlp", "gpt"):
            pool = [r for r in rows if r["seed"] == seed and r["kind"] == kind]
            if not pool or not any(r["context"] == context for r in pool):
                raise ValueError("동일 문맥/최대 성능 비교 후보가 누락됨")
            fixed.append(next(r["id"] for r in pool if r["context"] == context))
            best.append(min(pool, key=lambda r: (r["scores"]["all"]["ppl"], r["context"]))["id"])
    return {"matched_context": fixed, "best_in_declared_search": best}


def aggregate(rows, ids):
    selected = [r for r in rows if r["id"] in ids]
    result = {}
    for kind in ("count", "mlp", "gpt"):
        values = [r["scores"]["all"]["ppl"] for r in selected if r["kind"] == kind]
        result[kind] = {"seeds": len(values), "ppl_mean": statistics.mean(values),
                        "ppl_std": statistics.stdev(values) if len(values)>1 else None}
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def markdown(report):
    lines = ["# v0.3.5 공정 비교 실측", "", report["interpretation"], ""]
    by_id = {r["id"]: r for r in report["runs"]}
    for track, ids in report["tracks"].items():
        lines += ["## " + {"matched_context":"동일 3토큰 문맥 비교" if report["contract"]["novelty_context"]==3 else "동일 문맥 비교", "best_in_declared_search":"사전 선언한 후보 내 최선 성능"}[track], "", "| 모델 | 시드 | 문맥 | PPL | top-1 | top-5 | support | 규모 | 학습·저장 초 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for id in ids:
            r=by_id[id];m=r["scores"]["all"]
            lines.append(f'| {r["kind"]} | {r["seed"]} | {r["context"]} | {m["ppl"]:.4f} | {m["top1"]:.2%} | {m["top5"]:.2%} | {m["support"]:.2%} | {r["size"]:,} | {r["train_seconds"]:.1f} |')
    lines += ["", "각 계열의 전체 후보 학습·저장 시간(평가·생성 시간 제외): " + ", ".join(
        f"{kind} {sum(r['train_seconds'] for r in report['runs'] if r['kind']==kind):.1f}초"
        for kind in ('count','mlp','gpt')), ""]
    lines += ["", "## 신규성별 PPL (공통 판정 문맥)", "", "| 후보 | 본 조합 | 새 조합 | 새 문맥 |", "|---|---:|---:|---:|"]
    for r in report["runs"]:
        cells = [f'{v["ppl"]:.2f} (n={v["n"]})' if v["n"] else "N/A (n=0)" for v in r["scores"]["buckets"].values()]
        lines.append("| " + r["id"] + " | " + " | ".join(cells) + " |")
    lines += ["", "## 생성 예시 (greedy, 첫 고정 프롬프트)", ""]
    for r in report["runs"]:
        sample = r["generation"]["0.0"]["samples"][0]
        lines.append(f'- {r["id"]}: {sample["prompt"]} → {sample["continuation"] or "(생성 없음)"}')
    lines += ["", "세부 설정·데이터/가중치 SHA-256·학습량·에폭별 기록·온도 0.7 생성·다양성·시드 집계는 comparison_report.json 참조.", "카운트 규모는 저장된 빈도 셀 수이며 신경망의 학습 파라미터 수와 단위가 다릅니다."]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description="v0.3.5: 같은 데이터로 재학습하는 3자 캡스톤")
    parser.add_argument("--train", type=Path, default=Path(gpt.DATA_PATH))
    parser.add_argument("--valid", type=Path, default=Path(gpt.VALID_PATH))
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234])
    parser.add_argument("--context", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embed", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--threads", type=int, default=4)
    args=parser.parse_args()
    if min(args.context,args.epochs,args.batch_size,args.embed,args.hidden,args.layers,args.heads,args.threads)<1 or args.lr<=0 or args.patience<0:
        parser.error("설정값 범위를 확인하세요.")
    if args.embed % args.heads:
        parser.error("embed는 heads로 나누어져야 합니다.")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("시드는 중복될 수 없습니다.")
    torch.set_num_threads(args.threads)
    reader = gpt.Model()
    train, valid = reader.read_sentences(args.train), reader.read_sentences(args.valid)
    if not train or not valid:
        parser.error("학습/검증 데이터가 비어 있습니다.")
    if args.train.resolve() == args.valid.resolve() or sha(args.train) == sha(args.valid):
        parser.error("학습과 검증 파일은 달라야 합니다.")
    reader.itos = reader.build_vocab(train)
    reader.stoi = {t:i for i,t in enumerate(reader.itos)}
    labels = novelty(reader, train, valid, args.context)
    if not labels:
        parser.error("검증 채점 자리가 없습니다.")
    prompts = ["오늘은", "나는", "학교에서"]
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    runs=[]
    for seed in args.seeds:
        for kind,context in candidates(args.context):
            id=f"{kind}-c{context}-s{seed}"
            print(id,flush=True)
            destination=out/"candidates"/id/"0.model";destination.mkdir(parents=True,exist_ok=True)
            start=time.perf_counter()
            if kind=="count":
                lm=Count();lm.ORDERS=list(range(1,context+1));lm.BLOCK_SIZE=context
                lm.known=set(reader.itos[1:]);lm.stoi=reader.stoi.copy()
                tables=lm.train(train);lm.model=lm.to_dict(tables)
                path=destination/"model.json";lm.save(tables,path)
                size=sum(len(c) for t in tables.values() for c in t.values())
                history=[];best_epoch=None;epochs=1
                config={"orders":lm.ORDERS,"deterministic":True}
            else:
                lm=(MLP if kind=="mlp" else gpt.Model)()
                for k,v in {"DEVICE":"cpu","BLOCK_SIZE":context,"SEED":seed,"EPOCHS":args.epochs,
                            "PATIENCE":args.patience,"BATCH_SIZE":args.batch_size,"EMBED":args.embed,
                            "HIDDEN":args.hidden,"FFN_HIDDEN":args.hidden,"HEADS":args.heads,
                            "LAYERS":args.layers,"LR":args.lr}.items(): setattr(lm,k,v)
                lm.train(train,valid)
                path=destination/"model.pt";lm.save(path)
                size=sum(p.numel() for p in lm.net.parameters())
                history=lm.history;best_epoch=lm.best_epoch;epochs=len(history)
                config={k:getattr(lm,k) for k in ("EMBED","EPOCHS","PATIENCE","BATCH_SIZE","LR","SEED")}
                config.update({"optimizer":"Adam", "hidden":args.hidden} if kind=="mlp" else
                              {"optimizer":"Adam","layers":args.layers,"heads":args.heads,"ffn_hidden":args.hidden,"dropout":lm.DROPOUT})
            seconds=time.perf_counter()-start
            assert_alignment(reader,lm,train+valid)
            score=summarize(scored(lm,valid),labels)
            run={"id":id,"kind":kind,"seed":seed,"context":context,"config":config,
                 "scores":score,"train_scores":metrics(scored(lm,train)),"size":size,
                 "size_unit":"count_cells" if kind=="count" else "parameters",
                 "train_seconds":seconds,"epochs_run":epochs,"best_epoch":best_epoch,"history":history,
                 "training_target_tokens":len(positions(reader,train,context))*epochs,
                 "checkpoint":str(path.relative_to(out)),"checkpoint_sha256":sha(path),
                 "generation":generation_report(lm,prompts,seed)}
            runs.append(run)
            print(f'PPL {score["all"]["ppl"]:.4f}',flush=True)
    tracks=select(runs,args.context)
    # 첫 시드의 탐색 최저 GPT를 웹앱용 v0.3.5 산출물로 복사합니다. 시드 최저를 고르지 않습니다.
    chosen=next(r for r in runs if r["id"] in tracks["best_in_declared_search"] and r["kind"]=="gpt" and r["seed"]==args.seeds[0])
    exported=gpt.Model();exported.DEVICE="cpu";exported.load(out/chosen["checkpoint"]);exported.save(out/"model.pt")
    report={"version":"v0.3.5","contract":{"floor":FLOOR,"skip_first":True,"include_end":True,
        "count_probability":"normalized next_dist; no target-dependent backoff", "oov":"unknown target/context -> FLOOR and no rank credit",
        "novelty_context":args.context,"tokenizer":reader.tokenizer_name(),"tie_break":"vocabulary order", "prompts":prompts},
        "data":{name:{"sha256":sha(path),"sentences":len(sents)} for name,path,sents in [("train",args.train,train),("valid",args.valid,valid)]},
        "environment":{"python":platform.python_version(),"torch":torch.__version__,"device":"cpu","threads":args.threads},
        "candidate_grid":candidates(args.context),"runs":runs,"tracks":tracks,
        "aggregates":{k:aggregate(runs,ids) for k,ids in tracks.items()},"exported_gpt":chosen["id"],
        "interpretation":"동일 문맥 비교와 사전 선언한 후보 내 최고 검증 성능 비교입니다. 동일 계산량/파라미터 통제 실험이 아니며, 검증 세트로 선택·보고하므로 독립 테스트 성능 또는 보편적 최대 성능이 아닙니다. 카운트 채점 계약을 통일했으므로 이전 표의 PPL과 직접 동일시하지 않습니다."}
    (out/"comparison_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    (out/"comparison_report.md").write_text(markdown(report))
    print(out/"comparison_report.md")


if __name__=="__main__":
    main()
