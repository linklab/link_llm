"""Audit the SFT draft and legacy data without training or mutating either."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('sft050', ROOT/'data/sft/v0.5.0/dataset.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=d.HERE/'draft')
    parser.add_argument('--require-reviewed', action='store_true')
    args = parser.parse_args()
    manifest, splits = d.load_bundle(args.dataset, for_training=args.require_reviewed)
    print(json.dumps({'version': 'v0.5.0', 'mode': manifest['mode'],
                      'training_ready': manifest['training_ready'], 'splits': manifest['splits'],
                      'legacy_audit': d.legacy_audit(ROOT/'data/sft'),
                      'note': 'Structural audit only; no human review, model training or generation quality claim.'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
