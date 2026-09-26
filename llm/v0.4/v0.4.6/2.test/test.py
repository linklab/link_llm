"""Evaluate the frozen base model on sealed independent text; never select/tune here."""
import importlib.util
import math
from pathlib import Path
import torch

VERSION=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v046_evaluation',VERSION/'0.model/capstone.py')
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
m=c.m


def main():
    torch.set_num_threads(4)
    lm,freeze=c.verify()
    records,manifest=c.load_final_test(m.ROOT/'data/pretrain/v0.4.6',VERSION/'0.model/freeze.json')
    docs=[r['text'] for r in records]
    metrics=lm.evaluate(docs)
    _,old=m.corpus.load_bundle(m.ROOT/'data/pretrain/v0.4.1')
    prompts=freeze['generation_prompts']
    generations=[]
    for prompt in prompts:
        generated=lm.generate(prompt,seed=1234)
        uncached=lm.generate(prompt,use_cache=False,seed=1234)
        if generated!=uncached: raise ValueError('cached generation differs')
        suffix=generated[len(prompt):]
        generations.append({'prompt':prompt,'generated':generated,'continuation':suffix,
                            'only_whitespace_or_empty':not suffix.strip(),'replacement_characters':suffix.count('\ufffd')})
    targets=sum(len(lm.bpe.encode(t)) for t in docs)
    allowed=len(lm.itos)-5
    report={'version':lm.MODEL_VERSION,'freeze_sha256':m.corpus.sha256((VERSION/'0.model/freeze.json').read_bytes()),
            'weights_sha256':c.weights_hash(lm.net),'final_test_manifest':manifest,
            'independent_test':metrics,
            'topics':{topic:lm.evaluate([r['text'] for r in records if r['topic']==topic]) for topic in manifest['topics']},
            'legacy_test':{'note':'Previously exposed v0.4.1 test; diagnostic only, not the new independent test.',
                           'metrics':lm.evaluate([r['text'] for r in old['test']])},
            'uniform_baseline':{'ppl':allowed,'nll_per_byte':targets*math.log(allowed)/manifest['text_bytes'],
                                'note':'Equal probability over byte/BPE tokens plus EOS; text-byte NLL excludes EOS.'},
            'generation':generations,'quality':{'blank_continuations':sum(g['only_whitespace_or_empty'] for g in generations),
                'prompts':len(generations),'replacement_characters':sum(g['replacement_characters'] for g in generations)},
            'limitations':'12 authored synthetic documents, declared-group and normalized-duplicate isolation only. '
                          'Not a broad benchmark or a chat/instruction model. No tuning after final evaluation.'}
    c.validate_model(lm,freeze)
    (VERSION/'0.model/final_report.json').write_bytes(m.corpus.json_bytes(report))
    print({'independent_test':metrics,'quality':report['quality']})


if __name__=='__main__': main()
