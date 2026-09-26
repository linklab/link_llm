"""Freeze validation-selected weights and training budget BEFORE final-test creation."""
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import torch

VERSION=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v046_freeze_contract',VERSION/'0.model/capstone.py')
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
m=c.m


def main():
    torch.set_num_threads(4)
    output=VERSION/'0.model'
    if (output/'freeze.json').exists():
        c.verify(output); print('Existing freeze verified; no overwrite'); return
    if (m.ROOT/'data/pretrain/v0.4.6/test.jsonl').exists():
        raise ValueError('final test already exists: freeze must precede its creation')
    source=Path(m.previous.MODEL_PATH)
    selection=json.loads(source.with_name('architecture_report.json').read_text())
    source_hash=m.corpus.sha256(source.read_bytes())
    if source_hash!=selection['selected_checkpoint_sha256']:
        raise ValueError('selected source checkpoint changed')
    old=m.previous.Model(); old.DEVICE='cpu'; old.load(source)
    lm=m.Model(); lm.DEVICE='cpu'; lm.itos=old.itos; lm.stoi=old.stoi
    lm.BLOCK_SIZE=old.BLOCK_SIZE; lm.EMBED=old.EMBED; lm.best_epoch=old.best_epoch
    lm.restore_extra_metadata(old.extra_metadata())
    lm.net=lm.build_net(); lm.net.load_state_dict(old.net.state_dict()); lm.net.eval()
    assert c.weights_hash(lm.net)==c.weights_hash(old.net)
    lm.save(output/'model.pt')
    source_report=json.loads(source.with_name('training_report.json').read_text())
    freeze={'format':1,'version':lm.MODEL_VERSION,
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=m.ROOT,text=True).strip(),
            'source_checkpoint_sha256':source_hash,
            'export_checkpoint_sha256':m.corpus.sha256((output/'model.pt').read_bytes()),
            'weights_sha256':c.weights_hash(lm.net),'tokenizer_semantic_sha256':c.tokenizer_hash(lm),
            'architecture':{k:getattr(lm,k) for k in c.ARCH_KEYS},
            'training_config':lm.training_config(),'training_tokens':source_report['training_tokens'],
            'corpus_manifest_sha256':lm.corpus_manifest_sha256,
            'best_epoch':lm.best_epoch,'source_hashes':c.source_hashes(),
            'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),'threads':4,'device':'cpu'},
            'selection':selection['selection'],'selected_variant':selection['selected_variant'],
            'evaluation_protocol':'Freeze before final-test creation; new authored documents with disjoint '
                                  'declared topic/template groups and normalized duplicate checks. '
                                  'Report raw text/EOS NLL, PPL, NLL/byte, top1 and fixed generation prompts. '
                                  'Never tune on final test; previously exposed v0.4.1 test is diagnostic only.',
            'generation_prompts':['',' ','오늘은','처음 보는 글자 🐙','도자기 공방에서는','기록을 다시 읽으니'],
            'reproduction':'Same CPU runtime/config/data, exact semantic weight hash; serialization bytes are provenance only.'}
    (output/'freeze.json').write_bytes(m.corpus.json_bytes(freeze))
    report={**source_report,'version':lm.MODEL_VERSION,
            'checkpoint_sha256':freeze['export_checkpoint_sha256'],
            'preparation':'Frozen exact v0.4.5 selected weights; inherited training measurements, no extra optimization.'}
    (output/'training_report.json').write_bytes(m.corpus.json_bytes(report))
    c.verify(output)
    print('Frozen',freeze['selected_variant'],freeze['weights_sha256'])


if __name__=='__main__': main()
