"""Architecture-only inference costs on random weights; does not select a model."""
import importlib.util
from pathlib import Path
import statistics
import time
import torch

VERSION=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v045_cost',VERSION/'0.model/lm.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def main():
    torch.set_num_threads(4); rows=[]
    for variant in m.VARIANTS:
        torch.manual_seed(1234)
        net=m.ModernNetwork(768,64,64,4,256,.1,4,variant,2)
        net.head.weight=net.embedding.weight; net.eval()
        windows=[torch.tensor([[2]+[103]*(length-1)]) for length in range(16,49)]
        samples=[]
        for repeat in range(6):
            cache=None; start=time.perf_counter()
            for x in windows: _,cache=net.cached(x,cache)
            elapsed=time.perf_counter()-start
            if repeat: samples.append(elapsed)
        rows.append({'variant':variant,'median_seconds':statistics.median(samples),
                     'kv_bytes':sum(t.numel()*t.element_size() for pair in cache['layers'] for t in pair),
                     'params':sum(p.numel() for p in net.parameters())})
    report={'protocol':'CPU 4 threads, 5 repeats after warmup, 33 predictions including prefill, '
                        '48-token final cache. Random weights, same architecture as training; no quality claim.',
            'results':rows}
    (VERSION/'0.model/architecture_cost.json').write_bytes(m.corpus.json_bytes(report))
    print(report)


if __name__=='__main__': main()
