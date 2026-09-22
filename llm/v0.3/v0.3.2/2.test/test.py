"""저장 모델의 평가·생성 실행기. 구조 검증은 test_invariants.py."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

VERSION = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v032", VERSION / "0.model/lm.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=model.MODEL_PATH)
    parser.add_argument("--valid", default=model.VALID_PATH)
    parser.add_argument("--prompt", default="오늘은")
    args = parser.parse_args()
    torch.set_num_threads(4)
    lm = model.Model().load(args.model)
    print(json.dumps(lm.evaluate(lm.read_sentences(args.valid)), ensure_ascii=False, indent=2))
    print("생성:", lm.generate(args.prompt, temperature=0.0))


if __name__ == "__main__":
    main()
