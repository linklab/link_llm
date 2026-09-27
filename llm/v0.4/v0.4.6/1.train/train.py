"""Run local training experiments, or explicitly require strict reproduction."""
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
DEFAULT_LOCAL_OUTPUT=VERSION/'1.train/runs/local/0.model'
# Editable IDE defaults for local experiments; strict mode uses freeze.json.
PATIENCE=m.Model.PATIENCE
EPOCHS=m.Model.EPOCHS


def check_sources(frozen,strict):
    actual=c.source_hashes(frozen)
    if strict:
        c.validate_sources(frozen,actual)
    return actual


def training_config(freeze,strict,epochs=None,patience=None):
    config=dict(freeze['training_config'])
    for key,value,default in [('EPOCHS',epochs,EPOCHS),('PATIENCE',patience,PATIENCE)]:
        if strict:
            if value is not None and value!=config[key]:
                raise ValueError(f'--strict-reproduce requires {key}={config[key]}')
        else:
            config[key]=default if value is None else value
    if type(config['PATIENCE']) is not int or config['PATIENCE']<0 or config['EPOCHS']<1:
        raise ValueError('epochs must be positive and patience must be a nonnegative integer')
    return config


def runtime_info(threads):
    return {'python':platform.python_version(),'torch':str(torch.__version__),
            'threads':threads,'device':'cpu'}


def check_runtime(recorded,actual,strict):
    mismatches=[f'{key}: recorded={recorded[key]}, current={actual[key]}'
                for key in ('python','torch') if recorded[key]!=actual[key]]
    if strict and mismatches:
        raise ValueError('엄격한 재현 환경이 다릅니다. '+ '; '.join(mismatches)+
                         '. 일반 로컬 학습은 --strict-reproduce 없이 실행하세요.')
    return mismatches


def output_directory(requested,strict,temporary):
    target=Path(requested) if requested is not None else Path(temporary) if strict else DEFAULT_LOCAL_OUTPUT
    target=target.resolve()
    if not strict and (target==(VERSION/'0.model').resolve() or (target/'freeze.json').exists()):
        raise ValueError('일반 학습으로 고정 모델을 덮어쓸 수 없습니다. 별도 --output-dir을 사용하세요.')
    return target


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict-reproduce',action='store_true',
                        help='Require recorded Python/PyTorch and exact frozen weights (default: local training)')
    parser.add_argument('--output-dir',type=Path,
                        help='Default: 1.train/runs/local/0.model; strict mode without this option uses a temporary directory')
    parser.add_argument('--epochs',type=int,help='Local maximum epochs (strict mode requires the frozen value)')
    parser.add_argument('--patience',type=int,help='Stop after this many non-improving validation epochs; 0 disables early stopping')
    args=parser.parse_args(argv)
    freeze=json.loads((VERSION/'0.model/freeze.json').read_text())
    torch.set_num_threads(freeze['runtime']['threads'])
    actual_runtime=runtime_info(torch.get_num_threads())
    try:
        sources=check_sources(freeze['source_hashes'],args.strict_reproduce)
        config=training_config(freeze,args.strict_reproduce,args.epochs,args.patience)
        mismatches=check_runtime(freeze['runtime'],actual_runtime,args.strict_reproduce)
        # Check the output before spending time on training.
        output_directory(args.output_dir,args.strict_reproduce,tempfile.gettempdir())
    except ValueError as error:
        parser.error(str(error))
    print('모드: '+('엄격한 재현 검증' if args.strict_reproduce else '일반 로컬 학습'),flush=True)
    print(f"현재 Python {actual_runtime['python']} / PyTorch {actual_runtime['torch']}",flush=True)
    print(f"최대 에폭 {config['EPOCHS']} / PATIENCE {config['PATIENCE']} (0: 조기 종료 끄기)",flush=True)
    if mismatches:
        print('기록 환경과 다르므로 동일 가중치를 보장하지 않습니다: '+'; '.join(mismatches),flush=True)
    _,splits=m.corpus.load_bundle(m.ROOT/'data/pretrain/v0.4.1')
    manifest_hash=m.corpus.sha256((m.ROOT/'data/pretrain/v0.4.1/manifest.json').read_bytes())
    if manifest_hash!=freeze['corpus_manifest_sha256']: raise ValueError('corpus changed')
    lm=m.Model(); lm.DEVICE='cpu'; lm.corpus_manifest_sha256=manifest_hash
    for key,value in config.items(): setattr(lm,key,value)
    train,valid=([r['text'] for r in splits[k]] for k in ('train','valid'))
    start=time.perf_counter(); lm.train(train,valid); seconds=time.perf_counter()-start
    digest=c.weights_hash(lm.net)
    expected=freeze if args.strict_reproduce else {**freeze,'weights_sha256':digest}
    c.validate_model(lm,expected)
    if args.strict_reproduce and sum(s['tokens'] for s in lm.step_history)!=freeze['training_tokens']:
        raise ValueError('training budget changed')
    with tempfile.TemporaryDirectory(prefix='link-v046-reproduce-') as temporary:
        target=output_directory(args.output_dir,args.strict_reproduce,temporary)
        lm.save(target/'model.pt')
        restored=m.Model(); restored.DEVICE='cpu'; restored.load(target/'model.pt')
        c.validate_model(restored,expected)
        c.validate_sidecars(restored,target)
        metrics=restored.evaluate(valid)
        report={'version':lm.MODEL_VERSION,
                'status':'exact semantic weight/architecture/tokenizer match' if args.strict_reproduce else 'local training completed',
                'mode':'strict_reproduction' if args.strict_reproduce else 'local_training',
                'weights_sha256':digest,'matches_frozen_weights':digest==freeze['weights_sha256'],
                'training_tokens':sum(s['tokens'] for s in lm.step_history),'best_epoch':lm.best_epoch,
                'training_config':config,'seconds':seconds,'valid':metrics,
                'epochs_completed':len(lm.history),'early_stopped':lm.early_stopped,
                'history':lm.history,'source_hashes':sources,
                'runtime':actual_runtime,'recorded_runtime':freeze['runtime'],'runtime_differences':mismatches,
                'corpus_manifest_sha256':manifest_hash,
                'checkpoint_sha256':m.corpus.sha256((target/'model.pt').read_bytes()),
                'note':'Frozen architecture and corpus; local epochs/patience may differ. No final-test text read or scored. Local mode is not strict reproduction.'}
        name='reproduction_report.json' if args.strict_reproduce else 'training_report.json'
        (target/name).write_bytes(m.corpus.json_bytes(report))
        print(json.dumps(report,ensure_ascii=False,indent=2))
        if args.strict_reproduce and args.output_dir is None:
            print('재현 검증 완료. 임시 모델은 삭제하며 기존 모델은 유지합니다.')
        else:
            print(f'모델 저장: {target / "model.pt"}',flush=True)


if __name__=='__main__': main()
