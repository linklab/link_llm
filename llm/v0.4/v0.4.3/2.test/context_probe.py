"""Balanced two-choice recall diagnostic; reports failures without tuning the model."""
import importlib.util
from pathlib import Path
import torch

VERSION=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v043_probe',VERSION/'0.model/lm.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def score(lm,prompt,answer):
    ids=[2]+lm.bpe.encode(prompt)
    result=0.
    for target in lm.bpe.encode(answer):
        with torch.no_grad():
            logits=lm.net(torch.tensor([ids[-lm.BLOCK_SIZE:]],device=lm.device()))[0,-1]
        result+=float(logits.double().log_softmax(-1)[target])
        ids.append(target)
    return result


def main():
    torch.set_num_threads(4)
    lm=m.Model(); lm.DEVICE='cpu'; lm.load(m.MODEL_PATH)
    rows=[]
    for distance in (40,90):
        for color in ('파랑','노랑'):
            prefix=f'비밀 색: {color}. '
            while len(lm.bpe.encode(prefix))<distance:
                prefix+='바람. '
            prompt=prefix+'비밀 색:'
            scores={c:score(lm,prompt,' '+c) for c in ('파랑','노랑')}
            prediction=max(scores,key=scores.get)
            rows.append({'target':color,'prompt':prompt,'prefix_tokens':len(lm.bpe.encode(prompt))+1,
                         'context':lm.BLOCK_SIZE,'log_probabilities':scores,'prediction':prediction,
                         'correct':prediction==color})
    report={'protocol':'Frozen-model balanced color recall, full answer conditional log probability; '
                        '4 authored synthetic probes, no training or model selection; chance 50%. '
                        'Long prefixes intentionally put the clue outside the context window.',
            'checkpoint_sha256':m.corpus.sha256(Path(m.MODEL_PATH).read_bytes()),
            'accuracy':sum(r['correct'] for r in rows)/len(rows),'cases':rows}
    (VERSION/'0.model/context_probe.json').write_bytes(m.corpus.json_bytes(report))
    print(report)


if __name__=='__main__': main()
