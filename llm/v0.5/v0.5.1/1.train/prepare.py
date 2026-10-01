"""Export reversible chat token sequences; this is not an SFT training loop."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile

VERSION = Path(__file__).resolve().parents[1]
ROOT = VERSION.parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


chat = module('chat051_prepare', VERSION/'0.model/chat_template.py')
data = module('sft050_prepare051', ROOT/'data/sft/v0.5.0/dataset.py')
DEFAULT_TOKENIZER = ROOT/'llm/v0.4/v0.4.6/0.model/tokenizer.json'
DEFAULT_DATA = ROOT/'data/sft/v0.5.0/draft'


def load_prepared(directory, *, for_training=True):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_bytes())
    if manifest.get('format') != 'sft-chat-sequences-v1' or manifest.get('version') != 'v0.5.1':
        raise ValueError('unsupported prepared dataset format')
    if data.digest((directory/'source/manifest.json').read_bytes()) != manifest['source_manifest_sha256']:
        raise ValueError('source manifest hash mismatch')
    source, splits = data.load_bundle(directory/'source', for_training=for_training)
    if manifest.get('training_ready') is not source['training_ready']:
        raise ValueError('review readiness mismatch')
    if data.digest((directory/'template.json').read_bytes()) != manifest['template_sha256']:
        raise ValueError('template hash mismatch')
    template = chat.ChatTemplate.load(directory/'template.json')
    config = template.to_dict()
    for key in ('base_vocab_size', 'vocab_size', 'eot_id', 'base_tokenizer_sha256'):
        if manifest.get(key) != config[key]:
            raise ValueError('template metadata mismatch')
    limit = manifest.get('max_tokens_limit')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('invalid token limit')
    prepared = {}
    for name, records in splits.items():
        raw = (directory/f'{name}.jsonl').read_bytes()
        if data.digest(raw) != manifest['splits'][name]['sha256']:
            raise ValueError('sequence file hash mismatch')
        entries = data.read_jsonl(directory/f'{name}.jsonl')
        expected = [{'id': r['id'], **template.encode_conversation(r['messages'])} for r in records]
        if entries != expected:
            raise ValueError('encoded sequence differs from source conversation')
        if limit is not None and any(len(e['input_ids']) > limit for e in entries):
            raise ValueError('encoded sequence exceeds declared token limit')
        if statistics(entries) != manifest['splits'][name]['statistics']:
            raise ValueError('sequence statistics mismatch')
        prepared[name] = entries
    return manifest, template, splits, prepared


def statistics(entries):
    lengths = [len(e['input_ids']) for e in entries]
    return {'conversations': len(entries), 'tokens': sum(lengths),
            'min_tokens': min(lengths), 'max_tokens': max(lengths),
            'exceeds_64_tokens': sum(n > 64 for n in lengths)}


def prepare(dataset, tokenizer_path, output, *, allow_draft=False, max_tokens=None):
    dataset, tokenizer_path, output = (Path(p).resolve() for p in (dataset, tokenizer_path, output))
    if (output.is_relative_to(dataset) or dataset.is_relative_to(output)
            or tokenizer_path.is_relative_to(output)):
        raise ValueError('output must be separate from source data and tokenizer')
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
        raise ValueError('max_tokens must be a positive integer')
    source, splits = data.load_bundle(dataset, for_training=not allow_draft)
    template = chat.ChatTemplate(chat.ByteBPE.load(tokenizer_path))
    sequences = {name: [{'id': r['id'], **template.encode_conversation(r['messages'])}
                        for r in rows] for name, rows in splits.items()}
    too_long = [e['id'] for rows in sequences.values() for e in rows
                if max_tokens is not None and len(e['input_ids']) > max_tokens]
    if too_long:
        raise ValueError(f'{len(too_long)} conversations exceed max_tokens={max_tokens}; no truncation: {too_long[:3]}')
    source_files = {f'source/{name}': (dataset/name).read_bytes()
                    for name in ['manifest.json', 'train.jsonl', 'valid.jsonl', 'test.jsonl']}
    files = {**source_files, 'template.json': chat.json_bytes(template.to_dict())}
    manifest = {'format': 'sft-chat-sequences-v1', 'version': 'v0.5.1',
                'training_ready': source['training_ready'],
                'source_manifest_sha256': data.digest(files['source/manifest.json']),
                'template_sha256': data.digest(files['template.json']),
                'base_tokenizer_sha256': template.to_dict()['base_tokenizer_sha256'],
                'base_vocab_size': template.base_vocab_size, 'vocab_size': template.vocab_size,
                'eot_id': template.eot_id, 'max_tokens_limit': max_tokens,
                'note': 'No fitting, truncation, loss masking or model training. Source review/splits preserved. New EOT needs model vocabulary expansion before training.',
                'splits': {}}
    for name, entries in sequences.items():
        files[f'{name}.jsonl'] = data.jsonl(entries)
        manifest['splits'][name] = {'sha256': data.digest(files[f'{name}.jsonl']),
                                   'statistics': statistics(entries)}
    files['manifest.json'] = data.encode(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='chat051-', dir=output.parent) as temporary:
        staged = Path(temporary)/'export'
        for name, raw in files.items():
            path = staged/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        load_prepared(staged, for_training=not allow_draft)
        if output.exists():
            if not output.is_dir() or {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} != set(files):
                raise FileExistsError('output already exists; choose a new output directory')
            if any((output/name).read_bytes() != raw for name, raw in files.items()):
                raise FileExistsError('output differs; choose a new output directory')
        else:
            staged.rename(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--tokenizer', type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument('--output-dir', type=Path, default=VERSION/'1.train/preview')
    parser.add_argument('--allow-draft', action='store_true', help='Explicit inspection-only export of unreviewed data')
    parser.add_argument('--max-tokens', type=int, help='Reject oversized conversations; never truncate')
    args = parser.parse_args(argv)
    try:
        result = prepare(args.dataset, args.tokenizer, args.output_dir,
                         allow_draft=args.allow_draft, max_tokens=args.max_tokens)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
