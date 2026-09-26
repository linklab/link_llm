"""Predeclared one-at-a-time architecture experiment; validation-only selection."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

VERSION=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v045_compare',VERSION/'0.model/lm.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def main():
    output=VERSION/'0.model'; output.mkdir(parents=True,exist_ok=True)
    protocol={'version':'v0.4.5','variants':list(m.VARIANTS),
              'selection':'Lowest validation NLL/byte, ties by parameter count then variant name. Never test.',
              'config':{'context':64,'embed':64,'layers':4,'heads':4,'ffn_hidden':256,
                        'epochs':4,'seed':1234,'batch_size':128,'lr':.003,'device':'cpu','threads':4},
              'controls':'Same train-only tokenizer, data, epochs, optimizer/schedule and seed. '
                         'Only one architecture component replaced per candidate; parameter counts differ.',
              'results':[]}
    path=output/'architecture_report.json'
    path.write_bytes(m.corpus.json_bytes(protocol))  # Write the protocol before running candidates.
    with tempfile.TemporaryDirectory(prefix='link-v045-') as directory:
        best=None
        for variant in m.VARIANTS:
            target=Path(directory)/variant
            subprocess.run([sys.executable,str(VERSION/'1.train/train.py'),'--variant',variant,
                            '--device','cpu','--output-dir',str(target)],check=True)
            result=json.loads((target/'training_report.json').read_text())
            result['config']['output_dir']='temporary candidate directory'
            result['variant']=variant
            protocol['results'].append(result)
            key=(result['valid']['nll_per_byte'],result['params'],variant)
            if best is None or key<best:
                best=key
                for name in ('model.pt','vocab.json','tokenizer.json','config.json','training_state.pt'):
                    shutil.copyfile(target/name,output/name)
                selected={**result,'selection':'Selected by validation among five single-component variants; no test used.'}
                (output/'training_report.json').write_bytes(m.corpus.json_bytes(selected))
            path.write_bytes(m.corpus.json_bytes(protocol))
        protocol['selected_variant']=best[2]
        protocol['selected_checkpoint_sha256']=m.corpus.sha256((output/'model.pt').read_bytes())
        path.write_bytes(m.corpus.json_bytes(protocol))
        print('Selected',best[2],flush=True)


if __name__=='__main__': main()
