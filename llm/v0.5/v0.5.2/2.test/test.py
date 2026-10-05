"""Check exported labels against every approved assistant response."""
import argparse
import importlib.util
import json
from pathlib import Path

V = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verify052', V/'1.train/prepare.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def verify(directory):
    manifest, template, rows = p.load(directory)
    _, _, source, _ = p.p.load_prepared(Path(directory)/'source')
    responses = tokens = eos = 0
    for name, records in source.items():
        actual = {(r['conversation_id'],r['assistant_message_index']):r for r in rows[name]}
        for record in records:
            for i,msg in enumerate(record['messages']):
                if msg['role'] != 'assistant': continue
                row = actual[(record['id'],i)]
                final = i == len(record['messages'])-1
                target = template.bpe.encode(msg['content'])+[template.eot_id]+([1] if final else [])
                if [x for x in row['labels'] if x != -100] != target:
                    raise ValueError('response target mismatch')
                first = next(j for j,y in enumerate(row['labels']) if y != -100)
                history = [record['messages'][j] for j in row['kept_message_indices'][:-1]]
                if template.encode_prompt(history)['input_ids'] != row['input_ids'][:first+1]:
                    raise ValueError('generation prompt mismatch')
                responses += 1; tokens += len(target); eos += final
    return {'version':'v0.5.2','training_ready':True,
            'assistant_responses_verified':responses,'supervised_tokens':tokens,
            'eot_targets':responses,'eos_targets':eos,
            'context':manifest['context'],'overflow':manifest['overflow'],
            'splits':manifest['splits'],'model_trained':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=p.DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset),ensure_ascii=False,indent=2))
