"""학습 보고서의 원문 해시를 확인하고 BPE 전용 지표를 평가합니다."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import torch

VERSION=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v040',VERSION/'0.model/lm.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)


def main():
    parser=argparse.ArgumentParser(description='토큰 PPL은 이전 punct+josa와 직접 비교하지 않습니다.')
    parser.add_argument('--model',type=Path,default=Path(model.MODEL_PATH))
    parser.add_argument('--valid',type=Path,default=Path(model.VALID_PATH))
    parser.add_argument('--report',type=Path)
    parser.add_argument('--allow-new-data',action='store_true',help='보고서와 다른 원문을 명시적으로 평가')
    parser.add_argument('--prompt',default='오늘은')
    args=parser.parse_args();torch.set_num_threads(4)
    lm=model.Model().load(args.model)
    source=args.valid.read_bytes()
    documents=lm.documents_from_bytes(source)
    file_hash=hashlib.sha256(source).hexdigest()
    document_hash=model.tokenizer.corpus_hash(documents)
    if not args.allow_new_data:
        report=json.loads((args.report or args.model.with_name('training_report.json')).read_text(encoding='utf-8'))
        if (file_hash!=report['data']['valid']['sha256']
                or document_hash!=report['scored_documents']['valid']):
            raise ValueError('검증 원문이 학습 보고서와 다릅니다. 새 데이터 평가에는 --allow-new-data를 명시하세요.')
        tokenizer_hash=hashlib.sha256((json.dumps(lm.bpe.to_dict(),ensure_ascii=False,indent=2)+'\n').encode('utf-8')).hexdigest()
        if (lm.bpe.training_hash!=report['tokenizer']['training_hash']
                or tokenizer_hash!=report['tokenizer']['sha256']):
            raise ValueError('체크포인트 토크나이저의 학습 원문이 보고서와 다릅니다.')
    print(json.dumps({'file_sha256':file_hash,'document_sha256':document_hash,
                      'metrics':lm.evaluate(documents)},ensure_ascii=False,indent=2))
    print('생성:',lm.generate(args.prompt,temperature=0.0))


if __name__=='__main__':main()
