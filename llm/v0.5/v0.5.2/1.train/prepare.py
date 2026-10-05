"""Export auditable assistant-only SFT examples from approved v0.5.1 data."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile

VERSION = Path(__file__).resolve().parents[1]
PREVIOUS = VERSION.parent/'v0.5.1'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


p = module('prepare051_for052', PREVIOUS/'1.train/prepare.py')
sft = module('sft052', VERSION/'0.model/sft_data.py')
DEFAULT_SOURCE = PREVIOUS/'1.train/release/prepared'
DEFAULT_OUTPUT = VERSION/'1.train/prepared'
CONTRACT = 'Already shifted: logits[t] predicts labels[t]. One target assistant per example; body+EOT, final EOS only. Prompt/roles/history/PAD=-100.'


def stats(rows):
    return {'examples': len(rows), 'supervised_tokens': sum(r['supervised_tokens'] for r in rows),
            'max_input_tokens': max(len(r['input_ids']) for r in rows),
            'context_reduced_examples': sum(bool(r['dropped_message_indices']) for r in rows)}


def load(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_bytes())
    if (manifest.get('format') != 'assistant-loss-v1' or manifest.get('training_ready') is not True
            or manifest.get('version') != 'v0.5.2' or manifest.get('contract') != CONTRACT):
        raise ValueError('approved assistant-loss export required')
    _, template, splits, _ = p.load_prepared(directory/'source', for_training=True)
    if p.data.digest((directory/'source/manifest.json').read_bytes()) != manifest['source_sha256']:
        raise ValueError('source hash mismatch')
    rows = {}
    for name, records in splits.items():
        rows[name] = p.data.read_jsonl(directory/f'{name}.jsonl')
        expected = [e for r in records for e in sft.examples(template, r, manifest['context'], manifest['overflow'])]
        if rows[name] != expected:
            raise ValueError('SFT labels/context differ from approved source')
        if manifest['splits'][name] != {'sha256': p.data.digest((directory/f'{name}.jsonl').read_bytes()), **stats(expected)}:
            raise ValueError('split hash/statistics mismatch')
    return manifest, template, rows


def build(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT, context=256, overflow='reject'):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('source and output must not overlap')
    _, template, splits, _ = p.load_prepared(source, for_training=True)
    encoded = {name: [e for r in records for e in sft.examples(template, r, context, overflow)]
               for name, records in splits.items()}
    manifest = {'format': 'assistant-loss-v1', 'version': 'v0.5.2', 'training_ready': True,
                'context': context, 'overflow': overflow,
                'source_sha256': p.data.digest((source/'manifest.json').read_bytes()),
                'contract': CONTRACT,
                'splits': {}}
    files = {f'source/{f.relative_to(source)}': f.read_bytes() for f in source.rglob('*') if f.is_file()}
    for name, rows in encoded.items():
        files[f'{name}.jsonl'] = p.data.jsonl(rows)
        manifest['splits'][name] = {'sha256': p.data.digest(files[f'{name}.jsonl']), **stats(rows)}
    files['manifest.json'] = p.data.encode(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sft052-', dir=output.parent) as tmp:
        staged = Path(tmp)/'export'
        for name, raw in files.items():
            f = staged/name; f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(raw)
        load(staged)
        if output.exists():
            actual = {str(f.relative_to(output)): f.read_bytes() for f in output.rglob('*') if f.is_file()} if output.is_dir() else {}
            if actual != files:
                raise FileExistsError('output differs; choose a new output directory')
        else:
            staged.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--context', type=int, default=256)
    parser.add_argument('--overflow', choices=sft.POLICIES, default='reject')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = load(args.output_dir)[0] if args.check else build(args.source, args.output_dir, args.context, args.overflow)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
