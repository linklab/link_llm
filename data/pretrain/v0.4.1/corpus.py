"""Versioned UTF-8 JSONL corpus: provenance, conservative dedup and group splits.

Text is never normalized in storage. NFC + whitespace folding is only a duplicate
comparison key; semantic/near-duplicate detection is intentionally not claimed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import unicodedata

HERE = Path(__file__).resolve().parent
FORMAT = 'document-corpus-v1'
SPLITS = ('train', 'valid', 'test')


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def duplicate_key(text):
    return sha256(' '.join(unicodedata.normalize('NFC', text).split()).encode('utf-8'))


def documents_hash(documents):
    digest = hashlib.sha256()
    for text in documents:
        raw = text.encode('utf-8')
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n').encode('utf-8')


def read_jsonl(path):
    result = []
    # JSONL uses LF separators, not str.splitlines(): U+2028/U+0085 can occur
    # literally inside a valid JSON string and must remain part of the document.
    lines = Path(path).read_bytes().decode('utf-8').split('\n')
    if lines[-1] == '':
        lines.pop()
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f'blank JSONL record: {path}:{line_no}')
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'invalid JSONL: {path}:{line_no}') from exc
        if not isinstance(record, dict):
            raise ValueError('document must be an object')
        result.append(record)
    return result


def validate_records(records, sources):
    if not isinstance(sources, dict) or not sources:
        raise ValueError('source manifest required')
    for source in sources.values():
        for field in ('uri', 'license', 'usage', 'description'):
            if not isinstance(source.get(field), str) or not source[field].strip():
                raise ValueError(f'source {field} required')
    ids = set()
    for record in records:
        for field in ('id', 'source', 'topic', 'template_id'):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f'document {field} required')
        if not isinstance(record.get('text'), str):
            raise ValueError('document text must be a string (empty document is allowed)')
        record['text'].encode('utf-8')
        if record['source'] not in sources or record['id'] in ids:
            raise ValueError('unknown source or duplicate document id')
        ids.add(record['id'])


def prepare(records, sources, output_dir, seed=1234, group_by=('topic', 'template_id'),
            valid_fraction=.15, test_fraction=.15, provenance=None):
    records = list(records)
    validate_records(records, sources)
    records = sorted(records, key=lambda r: r['id'])
    if (not 0 < valid_fraction < 1 or not 0 < test_fraction < 1
            or valid_fraction + test_fraction >= 1):
        raise ValueError('positive held-out fractions must sum to less than one')
    if any(key not in ('topic', 'template_id') for key in group_by):
        raise ValueError('group_by must contain topic and/or template_id')
    # Union BEFORE dedup so that removing a duplicate cannot hide a grouping edge.
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    first = {}
    for i, record in enumerate(records):
        keys = [('duplicate', duplicate_key(record['text']))]
        keys += [(key, record[key]) for key in group_by]
        for key in keys:
            if key in first:
                union(i, first[key])
            else:
                first[key] = i
    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    if len(groups) < 3:
        raise ValueError('at least 3 independent groups required; refusing leaky fallback split')
    roots = sorted(groups)
    random.Random(seed).shuffle(roots)
    n_valid = max(1, round(len(roots)*valid_fraction))
    n_test = max(1, round(len(roots)*test_fraction))
    if n_valid+n_test >= len(roots):
        raise ValueError('not enough groups for a nonempty training split')
    assignments = {root: 'valid' if i < n_valid else 'test' if i < n_valid+n_test else 'train'
                   for i, root in enumerate(roots)}
    split_records = {name: [] for name in SPLITS}
    seen, removed = {}, []
    group_ids = {root: sha256(json_bytes([records[i]['id'] for i in members]))
                 for root, members in groups.items()}
    for i, record in enumerate(records):
        key = duplicate_key(record['text'])
        if key in seen:
            removed.append({'removed_id': record['id'], 'kept_id': seen[key],
                            'source': record['source'], 'topic': record['topic'],
                            'template_id': record['template_id'], 'duplicate_key': key})
            continue
        seen[key] = record['id']
        row = dict(record, text_sha256=sha256(record['text'].encode('utf-8')),
                   duplicate_key=key, group_id=group_ids[find(i)])
        split_records[assignments[find(i)]].append(row)
    manifest = {'format': FORMAT, 'seed': seed, 'sources': sources,
                'provenance': provenance or {},
                'deduplication': {'policy': 'NFC plus whitespace folding for keys only',
                                 'input_documents': len(records), 'removed': removed,
                                 'kept_documents': len(seen)},
                'split_policy': {'group_by': list(group_by), 'groups': len(groups),
                                 'requested_valid_group_fraction': valid_fraction,
                                 'requested_test_group_fraction': test_fraction,
                                 'method': 'seeded shuffle of connected components; ratios are groups, not documents'},
                'splits': {}}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in split_records.items():
        raw = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows).encode('utf-8')
        (output_dir/f'{name}.jsonl').write_bytes(raw)
        texts = [r['text'] for r in rows]
        manifest['splits'][name] = {'file': f'{name}.jsonl', 'sha256': sha256(raw),
            'documents_hash': documents_hash(texts), 'documents': len(rows),
            'text_bytes': sum(len(t.encode('utf-8')) for t in texts),
            'multiline_documents': sum('\n' in t for t in texts),
            'topics': sorted({r['topic'] for r in rows}),
            'templates': sorted({r['template_id'] for r in rows}),
            'groups': len({r['group_id'] for r in rows})}
    (output_dir/'manifest.json').write_bytes(json_bytes(manifest))
    load_bundle(output_dir)
    return manifest


def load_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_bytes())
    if manifest.get('format') != FORMAT:
        raise ValueError('unsupported corpus format')
    group_by = manifest['split_policy']['group_by']
    if any(k not in ('topic', 'template_id') for k in group_by):
        raise ValueError('invalid split policy')
    splits, seen_ids, seen_texts, owners = {}, set(), set(), {}
    for name in SPLITS:
        info = manifest['splits'][name]
        if info['file'] != f'{name}.jsonl':
            raise ValueError('split filename must be local and canonical')
        path = directory/info['file']
        if sha256(path.read_bytes()) != info['sha256']:
            raise ValueError(f'{name}: file hash mismatch')
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f'{name}: empty split')
        validate_records(rows, manifest['sources'])
        for row in rows:
            text_key = duplicate_key(row['text'])
            if (row['id'] in seen_ids or text_key in seen_texts
                    or row['text_sha256'] != sha256(row['text'].encode('utf-8'))
                    or row['duplicate_key'] != text_key):
                raise ValueError('duplicate document or corrupted text metadata')
            seen_ids.add(row['id'])
            seen_texts.add(text_key)
            for field in ('group_id', *group_by):
                value = row.get(field)
                if not isinstance(value, str) or not value:
                    raise ValueError('missing grouping metadata')
                key = (field, value)
                if key in owners and owners[key] != name:
                    raise ValueError(f'{field}: split leakage')
                owners[key] = name
        texts = [r['text'] for r in rows]
        if (documents_hash(texts) != info['documents_hash'] or len(rows) != info['documents']
                or sum(len(t.encode('utf-8')) for t in texts) != info['text_bytes']):
            raise ValueError('split statistics mismatch')
        splits[name] = rows
    return manifest, splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='JSONL: id,text,source,topic,template_id')
    parser.add_argument('--sources', type=Path, required=True, help='JSON source id → uri,license,usage,description')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--group-by', choices=['topic', 'template_id', 'both', 'document'], default='both')
    args = parser.parse_args()
    grouping = ('topic', 'template_id') if args.group_by == 'both' else (() if args.group_by == 'document' else (args.group_by,))
    report = prepare(read_jsonl(args.input), json.loads(args.sources.read_bytes()), args.output_dir,
                     args.seed, grouping, provenance={'input_sha256': sha256(args.input.read_bytes()),
                     'sources_sha256': sha256(args.sources.read_bytes())})
    print(json.dumps(report['splits'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
