"""Reproduce the frozen recipe from scratch and require the exact semantic weights."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import tempfile
import time
import torch

VERSION=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v046_reproduction',VERSION/'0.model/capstone.py')
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
m=c.m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,help='Optional reconstructed model destination; default temporary directory')
    args=parser.parse_args()
    freeze=json.loads((VERSION/'0.model/freeze.json').read_text())
    if c.source_hashes(freeze['source_hashes'])!=freeze['source_hashes']:
        raise ValueError('source changed after freeze')
    torch.set_num_threads(freeze['runtime']['threads'])
    if platform.python_version()!=freeze['runtime']['python'] or str(torch.__version__)!=freeze['runtime']['torch']:
        raise ValueError('strict reproduction requires the recorded Python/PyTorch versions')
    _,splits=m.corpus.load_bundle(m.ROOT/'data/pretrain/v0.4.1')
    manifest_hash=m.corpus.sha256((m.ROOT/'data/pretrain/v0.4.1/manifest.json').read_bytes())
    if manifest_hash!=freeze['corpus_manifest_sha256']: raise ValueError('corpus changed')
    lm=m.Model(); lm.DEVICE='cpu'; lm.corpus_manifest_sha256=manifest_hash
    for key,value in freeze['training_config'].items(): setattr(lm,key,value)
    train,valid=([r['text'] for r in splits[k]] for k in ('train','valid'))
    start=time.perf_counter(); lm.train(train,valid); seconds=time.perf_counter()-start
    c.validate_model(lm,freeze)
    if sum(s['tokens'] for s in lm.step_history)!=freeze['training_tokens']:
        raise ValueError('training budget changed')
    with tempfile.TemporaryDirectory(prefix='link-v046-reproduce-') as temporary:
        target=args.output_dir or Path(temporary)
        lm.save(target/'model.pt')
        restored=m.Model(); restored.DEVICE='cpu'; restored.load(target/'model.pt')
        c.validate_model(restored,freeze)
        metrics=restored.evaluate(valid)
    report={'status':'exact semantic weight/architecture/tokenizer match','weights_sha256':c.weights_hash(lm.net),
            'training_tokens':sum(s['tokens'] for s in lm.step_history),'best_epoch':lm.best_epoch,
            'seconds':seconds,'valid':metrics,'runtime':freeze['runtime'],
            'note':'Independent fresh initialization and full retraining; final test never read or scored here.'}
    (VERSION/'0.model/reproduction_report.json').write_bytes(m.corpus.json_bytes(report))
    print(report)


if __name__=='__main__': main()
