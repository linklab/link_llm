"""python llm/v0.3/v0.4.0/1.train/train.py --device cpu --epochs 30"""
import argparse
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import torch

VERSION = Path(__file__).resolve().parents[1]
ROOT = VERSION.parents[2]          # 저장소 루트 (llm/v0.3/vX.Y.Z -> 루트)


def repo_path(value):
    """경로를 **저장소 루트 기준 상대 경로**로 바꿔요.

    training_report.json 은 커밋되는 파일이라, 절대 경로가 들어가면
    다른 사람이 클론했을 때 맞지 않고 홈 디렉터리 구조까지 드러나요.
    저장소 밖 경로는 그대로 둡니다.
    """
    try:
        return str(Path(value).resolve().relative_to(ROOT))
    except ValueError:
        return str(value)
spec = importlib.util.spec_from_file_location("v040", VERSION / "0.model/lm.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def main():
    parser = argparse.ArgumentParser(description="v0.4.0 바이트 BPE와 평가 계약 학습")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--vocab-size", type=int, default=768)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--train", default=model.DATA_PATH)
    parser.add_argument("--valid", default=model.VALID_PATH)
    parser.add_argument("--output-dir", type=Path, default=VERSION / "0.model")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    lm = model.Model()
    for key in ("vocab_size", "layers", "ffn_hidden", "dropout", "heads", "epochs", "embed", "block_size", "batch_size", "lr", "patience", "seed", "device"):
        setattr(lm, key.upper(), getattr(args, key))
    source_data = {"train": Path(args.train).read_bytes(), "valid": Path(args.valid).read_bytes()}
    train = lm.documents_from_bytes(source_data["train"])
    valid = lm.documents_from_bytes(source_data["valid"])
    if not train or not valid:
        parser.error("학습/검증 파일에는 각각 최소 한 문장이 필요합니다.")
    if (Path(args.train).resolve() == Path(args.valid).resolve()
            or set(train).intersection(valid)):
        parser.error("학습/검증 원문이 같으면 안 됩니다.")
    data_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in source_data.items()}
    start = time.perf_counter()
    lm.train(train, valid)
    for name, path in (("train", args.train), ("valid", args.valid)):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != data_hashes[name]:
            raise RuntimeError("학습 중 원문 파일이 바뀌었습니다. 고정된 데이터로 다시 실행하세요.")
    lm.save(args.output_dir / "model.pt")
    report = {"version": "v0.4.0", "config": {k: repo_path(v) if k in ("train", "valid", "output_dir") else v
              for k, v in vars(args).items()}, "device": str(lm.device()),
              "python": platform.python_version(), "torch": torch.__version__,
              "data": {name: {"sentences": len(sentences),
                        "sha256": data_hashes[name]}
                       for name, path, sentences in [("train", args.train, train),
                                                      ("valid", args.valid, valid)]},
              "params": sum(p.numel() for p in lm.net.parameters()),
              "vocab_size": len(lm.itos), "best_epoch": lm.best_epoch,
              "history": lm.history, "train": lm.evaluate(train), "valid": lm.evaluate(valid),
              "elapsed_seconds": time.perf_counter() - start,
              "evaluation": {"contract": lm.EVALUATION_CONTRACT,
                  "text": "LF-separated UTF-8 documents; LF separators excluded; no strip/normalization",
                  "token_ppl": "all text tokens plus EOS, conditioned on BOS; no probability floor",
                  "nll_per_byte": "text-token NLL / original UTF-8 bytes; EOS excluded",
                  "comparison": "not directly comparable to punct+josa PPL"},
              "tokenizer": {"training_hash": lm.bpe.training_hash,
                            "actual_vocab_size": len(lm.itos), "requested_vocab_size": lm.VOCAB_SIZE,
                            "sha256": hashlib.sha256((args.output_dir / "tokenizer.json").read_bytes()).hexdigest()},
              "scored_documents": {"train": model.tokenizer.corpus_hash(train), "valid": model.tokenizer.corpus_hash(valid)},
              "checkpoint": "inference weights/config; not a resumable optimizer checkpoint"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("params", "best_epoch", "valid", "elapsed_seconds")},
                     ensure_ascii=False, indent=2))
    print(f"저장: {args.output_dir}")


if __name__ == "__main__":
    main()
