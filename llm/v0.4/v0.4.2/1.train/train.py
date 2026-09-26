"""Train on a verified JSONL corpus. Test text is audited, never fitted or scored here."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import resource
import sys
import time
import torch

VERSION = Path(__file__).resolve().parents[1]
ROOT = VERSION.parents[2]
spec = importlib.util.spec_from_file_location('v042', VERSION/'0.model/lm.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
corpus = model.module('v042_corpus_cli', ROOT/'data/pretrain/v0.4.1/corpus.py')


def display_path(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', type=Path, default=ROOT/'data/pretrain/v0.4.1')
    parser.add_argument('--output-dir', type=Path, default=VERSION/'0.model')
    for name, default in [('epochs',6), ('vocab-size',768), ('layers',4), ('heads',4),
                          ('ffn-hidden',256), ('embed',64), ('block-size',32),
                          ('batch-size',128), ('patience',0), ('seed',1234), ('threads',4)]:
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='cpu')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--grad-clip', type=float, default=1.)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    manifest, splits = corpus.load_bundle(args.corpus_dir)
    manifest_hash = corpus.sha256((args.corpus_dir/'manifest.json').read_bytes())
    train, valid = ([r['text'] for r in splits[name]] for name in ('train','valid'))
    lm = model.Model()
    for key, value in vars(args).items():
        if key not in ('corpus_dir','output_dir','threads'):
            setattr(lm, key.upper(), value)
    lm.corpus_manifest_sha256 = manifest_hash
    start = time.perf_counter()
    lm.train(train, valid, resume=args.resume, max_steps=args.max_steps,
             checkpoint=args.output_dir/'training_state.pt')
    if not lm.completed:
        print('Paused: resume with the same configuration and --resume training_state.pt')
        return
    training_seconds = time.perf_counter()-start
    # Detect changes to any split or metadata during training before publishing weights.
    corpus.load_bundle(args.corpus_dir)
    if corpus.sha256((args.corpus_dir/'manifest.json').read_bytes()) != manifest_hash:
        raise RuntimeError('corpus manifest changed during training')
    lm.save(args.output_dir/'model.pt')
    report = {'version': lm.MODEL_VERSION,
              'config': {k: display_path(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                              'device': str(lm.device())},
              'corpus_manifest_sha256': manifest_hash,
              'data': manifest['splits'], 'split_policy': manifest['split_policy'],
              'params': sum(p.numel() for p in lm.net.parameters()),
              'vocab_size': len(lm.itos), 'best_epoch': lm.best_epoch, 'history': lm.history,
              'packing': lm.packing_stats, 'train': lm.evaluate(train), 'valid': lm.evaluate(valid),
              'test': {'status': 'not evaluated; use explicit --split test after model selection'},
              'tokenizer': {'training_hash': lm.bpe.training_hash,
                            'sha256': corpus.sha256((args.output_dir/'tokenizer.json').read_bytes())},
              'checkpoint_sha256': corpus.sha256((args.output_dir/'model.pt').read_bytes()),
              'training_seconds': training_seconds,
              'elapsed_seconds': time.perf_counter()-start,
              'evaluation_contract': lm.EVALUATION_CONTRACT,
              'comparison': 'New held-out topic/template splits; do not rank against v0.4.0 old validation.',
              'checkpoint': 'training_state.pt: full resume; model.pt: best validation inference',
              'steps': lm.step_history,
              'training_tokens': sum(s['tokens'] for s in lm.step_history),
              'targets_per_second': lm.invocation_tokens/training_seconds,
              'invocation_training_tokens': lm.invocation_tokens,
              'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)}
    (args.output_dir/'training_report.json').write_bytes(corpus.json_bytes(report))
    print(json.dumps({k: report[k] for k in ('params','best_epoch','valid','packing','elapsed_seconds')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
