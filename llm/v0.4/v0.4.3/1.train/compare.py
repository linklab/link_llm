"""Small CPU scaling experiment; change one axis per adjacent candidate."""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import torch

VERSION = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v043_comparison', VERSION/'0.model/lm.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=VERSION/'0.model/scaling_report.json')
    args=parser.parse_args()
    torch.set_num_threads(4)
    _, splits=m.corpus.load_bundle(m.ROOT/'data/pretrain/v0.4.1')
    train,valid=([r['text'] for r in splits[k]] for k in ('train','valid'))
    cases=[('baseline',32,64,4,2,False),('tied',32,64,4,2,True),
           ('context',64,64,4,2,True),('width',64,96,4,2,True),
           ('depth',64,96,6,2,True),('tokens',64,96,6,4,True)]
    report={'protocol':'Adjacent candidates change one axis; same train/valid, BPE, seed and optimizer. '
                        'Packing changes optimizer step count when context changes. No test or selection.',
            'device':'cpu','threads':4,'results':[]}
    for name,context,width,depth,epochs,tied in cases:
        lm=m.Model(); lm.DEVICE='cpu'; lm.BLOCK_SIZE=context; lm.EMBED=width
        lm.FFN_HIDDEN=256  # Keep FFN hidden width fixed to isolate embedding width.
        lm.LAYERS=depth; lm.EPOCHS=epochs; lm.TIE_WEIGHTS=tied
        start=time.perf_counter(); lm.train(train,valid); seconds=time.perf_counter()-start
        # A retrieval diagnostic, not trained task accuracy: score the known answer after a distractor.
        probe='기록: 비밀 색은 파랑입니다. '+('산책하며 바람 소리를 들었습니다. '*6)+'비밀 색은'
        tokens=[lm.BOS]+lm.tokenize(probe)
        context_ids=lm._context_ids(tokens)
        answer=lm.bpe.encode(' 파랑')[0]
        with torch.no_grad():
            logits=lm.net(torch.tensor([context_ids]))[0,-1]
            rank=int((logits>logits[answer]).sum())+1
        report['results'].append({'case':name,'context':context,'embed':width,'layers':depth,
            'ffn_hidden':256,'epochs':epochs,'tied':tied,'params':sum(p.numel() for p in lm.net.parameters()),
            'parameter_bytes':sum(p.numel()*p.element_size() for p in lm.net.parameters()),
            'tokens':sum(s['tokens'] for s in lm.step_history),'steps':len(lm.step_history),
            'seconds':seconds,'targets_per_second':lm.invocation_tokens/seconds,
            'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            'valid':lm.evaluate(valid),'probe_first_answer_token_rank':rank,
            'probe_prefix_tokens':len(tokens),'probe_note':'Single diagnostic; answer source may be outside window. Not retrieval accuracy.'})
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_bytes(m.corpus.json_bytes(report))
        print(name,report['results'][-1]['valid']['ppl'],flush=True)


if __name__=='__main__': main()
