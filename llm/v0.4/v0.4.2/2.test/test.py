"""Evaluate the frozen checkpoint on validation (default) or an explicit held-out test."""
import argparse
import importlib.util
import json
from pathlib import Path
import torch

VERSION = Path(__file__).resolve().parents[1]
ROOT = VERSION.parents[2]
spec = importlib.util.spec_from_file_location('v042_eval', VERSION/'0.model/lm.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
corpus = model.module('v042_eval_corpus', ROOT/'data/pretrain/v0.4.1/corpus.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(model.MODEL_PATH))
    parser.add_argument('--corpus-dir', type=Path, default=ROOT/'data/pretrain/v0.4.1')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--split', choices=['valid','test'], default='valid')
    parser.add_argument('--allow-new-data', action='store_true')
    parser.add_argument('--output-report', type=Path)
    parser.add_argument('--prompt', default='오늘은')
    parser.add_argument('--device', choices=['cpu','auto','mps','cuda'], default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest, splits = corpus.load_bundle(args.corpus_dir)
    digest = corpus.sha256((args.corpus_dir/'manifest.json').read_bytes())
    lm = model.Model(); lm.DEVICE = args.device; lm.load(args.model)
    report = json.loads((args.report or args.model.with_name('training_report.json')).read_bytes())
    checkpoint_hash = corpus.sha256(args.model.read_bytes())
    if checkpoint_hash != report['checkpoint_sha256']:
        raise ValueError('checkpoint does not match training report')
    if (lm.corpus_manifest_sha256 != report['corpus_manifest_sha256']
            or lm.bpe.training_hash != report['tokenizer']['training_hash']):
        raise ValueError('checkpoint provenance does not match training report')
    if not args.allow_new_data and digest != lm.corpus_manifest_sha256:
        raise ValueError('different corpus; explicitly pass --allow-new-data')
    records = splits[args.split]
    texts = [r['text'] for r in records]
    # One forward evaluation per document set; groups are additional diagnostics.
    long_texts = [t for t in texts if len(lm.bpe.encode(t))+1 > lm.BLOCK_SIZE]
    result = {'version': lm.MODEL_VERSION, 'split': args.split,
              'corpus_manifest_sha256': digest, 'checkpoint_sha256': checkpoint_hash,
              'documents_hash': manifest['splits'][args.split]['documents_hash'],
              'metrics': lm.evaluate(texts),
              'long_documents': lm.evaluate(long_texts) if long_texts else None,
              'topics': {topic: lm.evaluate([r['text'] for r in records if r['topic']==topic])
                         for topic in sorted({r['topic'] for r in records})},
              'prompt': args.prompt, 'generated': lm.generate(args.prompt),
              'note': 'test is for frozen-checkpoint reporting, never early stopping or tokenizer fitting'}
    if args.output_report:
        protected = {args.model.resolve(), (args.report or args.model.with_name('training_report.json')).resolve()}
        protected.update(p.resolve() for p in args.corpus_dir.iterdir() if p.is_file())
        protected.update((args.model.parent/name).resolve() for name in
                         ('tokenizer.json','config.json','vocab.json','training_report.json'))
        if args.output_report.resolve() in protected:
            raise ValueError('output-report must not overwrite corpus or checkpoint artifacts; choose a new path')
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_bytes(corpus.json_bytes(result))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
