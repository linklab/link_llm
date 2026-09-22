"""저장된 후보와 데이터 해시를 확인하고 보고서 PPL을 재현합니다."""
import argparse
import importlib.util
import json
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('v035_compare',Path(__file__).resolve().parents[1]/'0.model/comparison.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,default=m.HERE/'comparison_report.json')
    parser.add_argument('--train',type=Path,default=Path(m.gpt.DATA_PATH))
    parser.add_argument('--valid',type=Path,default=Path(m.gpt.VALID_PATH))
    args=parser.parse_args();report=json.loads(args.report.read_text());torch.set_num_threads(4)
    reader=m.gpt.Model();train=reader.read_sentences(args.train);valid=reader.read_sentences(args.valid);reader.itos=reader.build_vocab(train);reader.stoi={t:i for i,t in enumerate(reader.itos)}
    for name,path in [('train',args.train),('valid',args.valid)]:
        if m.sha(path)!=report['data'][name]['sha256']:raise ValueError(f'{name} 데이터 해시 불일치')
    labels=m.novelty(reader,train,valid,report['contract']['novelty_context'])
    for run in report['runs']:
        path=args.report.parent/run['checkpoint']
        if m.sha(path)!=run['checkpoint_sha256']:raise ValueError('체크포인트 해시 불일치')
        if run['kind']=='count':
            lm=m.Count().load(path);lm.BLOCK_SIZE=run['context'];lm.known=set(reader.itos[1:]);lm.stoi=reader.stoi
        else:
            lm=(m.MLP if run['kind']=='mlp' else m.gpt.Model)();lm.DEVICE='cpu';lm.load(path)
        m.assert_alignment(reader,lm,valid)
        actual=m.summarize(m.scored(lm,valid),labels)
        for group in ['all', *m.BUCKETS]:
            left=actual['all'] if group=='all' else actual['buckets'][group]
            right=run['scores']['all'] if group=='all' else run['scores']['buckets'][group]
            for key,value in left.items():
                reference=right[key]
                if value is None or reference is None:
                    if value!=reference:raise AssertionError((run['id'],group,key))
                elif abs(value-reference)>1e-5:raise AssertionError((run['id'],group,key))
        generated=m.generation_report(lm,report['contract']['prompts'],run['seed'])
        if generated!=run['generation']:raise AssertionError((run['id'],'generation'))
        print(run['id'],actual['all'])
    print('모든 후보의 저장본 PPL 재현 완료')


if __name__=='__main__':main()
