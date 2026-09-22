"""python llm/v0.3/v0.3.2/1.train/train.py --device cpu --epochs 60"""
import argparse
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import torch

VERSION = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v032", VERSION / "0.model/lm.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def main():
    parser = argparse.ArgumentParser(description="v0.3.2 학습형 위치 임베딩 학습")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--embed", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--train", default=model.DATA_PATH)
    parser.add_argument("--valid", default=model.VALID_PATH)
    parser.add_argument("--output-dir", type=Path, default=VERSION / "0.model")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    lm = model.Model()
    for key in ("heads", "epochs", "embed", "block_size", "batch_size", "lr", "patience", "seed", "device"):
        setattr(lm, key.upper(), getattr(args, key))
    train = lm.read_sentences(args.train)
    valid = lm.read_sentences(args.valid)
    if not train or not valid:
        parser.error("학습/검증 파일에는 각각 최소 한 문장이 필요합니다.")
    start = time.perf_counter()
    lm.train(train, valid)
    lm.save(args.output_dir / "model.pt")
    report = {"version": "v0.3.2", "config": {k: str(v) if isinstance(v, Path) else v
              for k, v in vars(args).items()}, "device": str(lm.device()),
              "python": platform.python_version(), "torch": torch.__version__,
              "data": {name: {"sentences": len(sentences),
                        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                       for name, path, sentences in [("train", args.train, train),
                                                      ("valid", args.valid, valid)]},
              "params": sum(p.numel() for p in lm.net.parameters()),
              "vocab_size": len(lm.itos), "best_epoch": lm.best_epoch,
              "history": lm.history, "train": lm.evaluate(train), "valid": lm.evaluate(valid),
              "elapsed_seconds": time.perf_counter() - start,
              "evaluation": "FLOOR=1e-4; skip first token; include END; exact sliding context",
              "checkpoint": "inference weights/config; not a resumable optimizer checkpoint"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("params", "best_epoch", "valid", "elapsed_seconds")},
                     ensure_ascii=False, indent=2))
    print(f"저장: {args.output_dir}")


if __name__ == "__main__":
    main()
