"""Bind the untouched final-test texts to the already frozen model."""
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('v046_seal',ROOT/'llm/v0.4/v0.4.6/0.model/capstone.py')
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


def main():
    freeze=ROOT/'llm/v0.4/v0.4.6/0.model/freeze.json'
    c.verify(freeze.parent)
    if (HERE/'manifest.json').exists():
        c.load_final_test(HERE,freeze); print('Existing evaluation seal verified'); return
    records=c.m.corpus.read_jsonl(HERE/'test.jsonl')
    _,splits=c.m.corpus.load_bundle(ROOT/'data/pretrain/v0.4.1')
    c.check_independence(records,[r for rows in splits.values() for r in rows])
    manifest={'format':1,'purpose':'independent final evaluation only; never training/tokenizer/selection',
              'source':'authored-v046-final','license':'MIT',
              'provenance':'12 AI-assisted texts individually authored after the v0.4.6 model freeze; '
                           'no external corpus. Six declared topics, two distinct texts each. '
                           'Independence means no fitting/selection exposure, not semantic dissimilarity.',
              'documents':len(records),'topics':sorted({r['topic'] for r in records}),
              'text_bytes':sum(len(r['text'].encode()) for r in records),
              'freeze_sha256':c.m.corpus.sha256(freeze.read_bytes()),
              'file_sha256':c.m.corpus.sha256((HERE/'test.jsonl').read_bytes())}
    (HERE/'manifest.json').write_bytes(c.m.corpus.json_bytes(manifest))
    c.load_final_test(HERE,freeze)
    print('Sealed independent test:',manifest['documents'],'documents')


if __name__=='__main__': main()
