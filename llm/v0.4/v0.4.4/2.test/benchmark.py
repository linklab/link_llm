"""Median CPU cache benchmark, including the sliding-window rebuild case."""
import importlib.util
from pathlib import Path
import statistics
import time
import torch

VERSION=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v044_benchmark',VERSION/'0.model/lm.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def main():
    torch.set_num_threads(4)
    lm=m.Model(); lm.DEVICE='cpu'; lm.load(m.MODEL_PATH)
    rows=[]
    for label,start,steps in [('within_window',16,32),('window_overflow',64,16)]:
        ids=[2]+[6+(i%256) for i in range(start+steps)]
        windows=[torch.tensor([ids[max(0,n-lm.BLOCK_SIZE):n]]) for n in range(start,start+steps)]
        cache=None; max_error=0.
        for x in windows:
            with torch.no_grad(): full=lm.net(x)[:,-1]
            cached,cache=lm.net.cached(x,cache)
            torch.testing.assert_close(full,cached,rtol=2e-5,atol=2e-6)
            finite=torch.isfinite(full)
            max_error=max(max_error,float((full[finite]-cached[finite]).abs().max()))
        timings={}
        for enabled in (False,True):
            samples=[]
            for repeat in range(6):
                cache=None; start_time=time.perf_counter()
                for x in windows:
                    with torch.no_grad():
                        if enabled: _,cache=lm.net.cached(x,cache)
                        else: lm.net(x)
                elapsed=time.perf_counter()-start_time
                if repeat: samples.append(elapsed)
            timings['cached' if enabled else 'full']=statistics.median(samples)
        kv_bytes=sum(t.numel()*t.element_size() for pair in cache['layers'] for t in pair)
        rows.append({'case':label,'predictions':steps,'median_seconds':timings,
                     'speedup':timings['full']/timings['cached'],'kv_tensor_bytes':kv_bytes,
                     'max_absolute_logit_error':max_error})
    report={'device':'cpu','threads':4,'repeats':5,'includes_prefill':True,
            'checkpoint_sha256':m.corpus.sha256(Path(m.MODEL_PATH).read_bytes()),
            'protocol':'Fixed token sequence, no sampling/EOS. Medians after one warmup. '
                       'KV tensor bytes exclude Python objects and transient attention buffers.',
            'results':rows}
    (VERSION/'0.model/cache_report.json').write_bytes(m.corpus.json_bytes(report))
    print(report)


if __name__=='__main__': main()
