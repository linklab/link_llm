"""Semantic weight/config fingerprints and independent evaluation-set sealing."""
import hashlib
import importlib.util
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v046_contract_model',HERE/'lm.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ROOT=m.ROOT
ARCH_KEYS=('EMBED','BLOCK_SIZE','HEADS','LAYERS','FFN_HIDDEN','DROPOUT','TIE_WEIGHTS','VARIANT','KV_HEADS')


def weights_hash(net):
    digest=hashlib.sha256()
    for name,tensor in sorted(net.state_dict().items()):
        t=tensor.detach().cpu().contiguous()
        header=json.dumps([name,str(t.dtype),list(t.shape)],separators=(',',':')).encode()
        raw=bytes(t.view(torch.uint8).reshape(-1).tolist())
        digest.update(len(header).to_bytes(8,'big')); digest.update(header)
        digest.update(len(raw).to_bytes(8,'big')); digest.update(raw)
    return digest.hexdigest()


def tokenizer_hash(lm):
    return m.corpus.sha256(m.corpus.json_bytes(lm.bpe.to_dict()))


def source_hashes(paths=None):
    if paths is None:
        files=[p for p in (ROOT/'llm').glob('**/0.model/*.py') if p.name!='capstone.py']
        files += [ROOT/'data/dataloader.py', ROOT/'data/pretrain/v0.4.1/corpus.py']
    else:
        # Future versions may add files; verify only the already frozen source inventory.
        files=[ROOT/path for path in paths]
    return {str(p.relative_to(ROOT)):m.corpus.sha256(p.read_bytes()) for p in sorted(files)}


def validate_model(lm,freeze):
    if (lm.MODEL_VERSION!='v0.4.6' or weights_hash(lm.net)!=freeze['weights_sha256']
            or tokenizer_hash(lm)!=freeze['tokenizer_semantic_sha256']
            or {k:getattr(lm,k) for k in ARCH_KEYS}!=freeze['architecture']
            or lm.corpus_manifest_sha256!=freeze['corpus_manifest_sha256']):
        raise ValueError('frozen model/architecture/tokenizer/data mismatch')


def validate_sidecars(lm,directory):
    directory=Path(directory)
    tokenizer=m.ByteBPE.load(directory/'tokenizer.json')
    config=json.loads((directory/'config.json').read_text())
    expected={'version':lm.MODEL_VERSION,'evaluation_contract':lm.EVALUATION_CONTRACT,
              'block_size':lm.BLOCK_SIZE,'embed':lm.EMBED,**lm.extra_metadata()}
    if tokenizer.to_dict()!=lm.bpe.to_dict() or config!=expected:
        raise ValueError('exported tokenizer/config mismatch')


def verify(directory=HERE):
    directory=Path(directory)
    freeze=json.loads((directory/'freeze.json').read_text())
    if freeze['source_hashes']!=source_hashes(freeze['source_hashes']):
        raise ValueError('model implementation changed after freeze')
    m.corpus.load_bundle(ROOT/'data/pretrain/v0.4.1')
    if m.corpus.sha256((ROOT/'data/pretrain/v0.4.1/manifest.json').read_bytes())!=freeze['corpus_manifest_sha256']:
        raise ValueError('training corpus changed after freeze')
    lm=m.Model(); lm.DEVICE='cpu'; lm.load(directory/'model.pt')
    validate_model(lm,freeze)
    validate_sidecars(lm,directory)
    return lm,freeze


def check_independence(records,old_records):
    if not records:
        raise ValueError('empty final test')
    ids,keys=set(),set()
    old_keys={m.corpus.duplicate_key(r['text']) for r in old_records}
    old_topics={r['topic'] for r in old_records}
    old_templates={r['template_id'] for r in old_records}
    for record in records:
        if any(not isinstance(record.get(k),str) or not record[k] for k in ('id','text','source','topic','template_id')):
            raise ValueError('invalid final record')
        key=m.corpus.duplicate_key(record['text'])
        if (record['id'] in ids or key in keys or key in old_keys
                or record['topic'] in old_topics or record['template_id'] in old_templates):
            raise ValueError('final test duplicate/topic/template overlap')
        ids.add(record['id']); keys.add(key)


def load_final_test(data_dir,freeze_path):
    data_dir=Path(data_dir)
    manifest=json.loads((data_dir/'manifest.json').read_text())
    if (manifest['freeze_sha256']!=m.corpus.sha256(Path(freeze_path).read_bytes())
            or manifest['file_sha256']!=m.corpus.sha256((data_dir/'test.jsonl').read_bytes())):
        raise ValueError('final test seal mismatch')
    records=m.corpus.read_jsonl(data_dir/'test.jsonl')
    _,splits=m.corpus.load_bundle(ROOT/'data/pretrain/v0.4.1')
    check_independence(records,[r for rows in splits.values() for r in rows])
    if len(records)!=manifest['documents']:
        raise ValueError('final test count mismatch')
    return records,manifest
