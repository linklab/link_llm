"""SFT data contracts and reproducible draft/reviewed exports; no torch required.

Messages are raw Unicode, not serialized role tokens. Human review is an explicit
content-bound receipt, never inferred from an automatic validation result.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

HERE = Path(__file__).resolve().parent
FORMAT = 'sft-conversations-v1'
SPLITS = ('train', 'valid', 'test')
TASKS = ('format', 'extract', 'calculate', 'summarize', 'rewrite',
         'multiturn', 'uncertainty', 'safety')
CRITERIA = ('instruction_following', 'correctness', 'naturalness',
            'context_consistency', 'safety')


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode('utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_jsonl(path):
    lines = Path(path).read_bytes().decode('utf-8').split('\n')
    if lines[-1] == '':
        lines.pop()
    rows = []
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f'{path}:{number}: invalid JSONL') from error
        if not isinstance(row, dict):
            raise ValueError('JSONL records must be objects')
        rows.append(row)
    return rows


def jsonl(rows):
    return ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows).encode('utf-8')


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def content_hash(row):
    # Bind approval to all source content, including task/template/rationale.
    return digest(encode(row))


def conversation_key(row):
    return digest(encode([(m['role'], normalized(m['content'])) for m in row['messages']]))


def user_keys(row):
    # Include every user turn, not just the first turn of a multi-turn dialogue.
    return {normalized(m['content']) for m in row['messages'] if m['role'] == 'user'}


def validate_records(rows, sources):
    if not isinstance(sources, dict) or not sources:
        raise ValueError('source registry required')
    for source in sources.values():
        if not isinstance(source, dict):
            raise ValueError('source entry must be an object')
        for field in ('uri', 'license', 'usage', 'authorship'):
            if not isinstance(source.get(field), str) or not source[field].strip():
                raise ValueError(f'source {field} required')
    ids = set()
    for row in rows:
        for field in ('id', 'source', 'template_id', 'rationale'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'{field} required')
        if row['id'] in ids or row['source'] not in sources or row.get('task') not in TASKS:
            raise ValueError('duplicate id, unknown source or invalid task')
        ids.add(row['id'])
        messages = row.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ValueError('messages required')
        expected = 'user'
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or set(msg) != {'role', 'content'}:
                raise ValueError('message requires only role and content')
            text = msg['content']
            if not isinstance(text, str) or not text.strip() or '\ufffd' in text:
                raise ValueError('empty or damaged message')
            try:
                text.encode('utf-8')
            except UnicodeEncodeError as error:
                raise ValueError('invalid Unicode') from error
            if any(ord(c) < 32 and c not in '\n\r\t' for c in text):
                raise ValueError('control character in message')
            if i == 0 and msg['role'] == 'system':
                continue
            if msg['role'] != expected:
                raise ValueError('roles must alternate user/assistant after optional system')
            expected = 'assistant' if expected == 'user' else 'user'
        if messages[-1]['role'] != 'assistant' or not user_keys(row):
            raise ValueError('conversation must end with an assistant answer')


def review_index(rows, reviews):
    by_id = {r['id']: r for r in rows}
    result = {}
    for review in reviews:
        key = review.get('id')
        if key not in by_id or key in result:
            raise ValueError('unknown or duplicate review id')
        if review.get('content_sha256') != content_hash(by_id[key]):
            raise ValueError(f'{key}: stale review; review the edited content again')
        if (review.get('decision') not in ('approved', 'rejected')
                or review.get('reviewer_kind') != 'human'
                or not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip()
                or not isinstance(review.get('note'), str) or not review['note'].strip()):
            raise ValueError('human review identity, decision and note required')
        if review['decision'] == 'approved' and (
                not isinstance(review.get('checks'), dict)
                or set(review['checks']) != set(CRITERIA)
                or any(value is not True for value in review['checks'].values())):
            raise ValueError('approved review requires all quality criteria')
        result[key] = review
    return result


def assign_groups(rows, seed):
    """Union prompt/template/duplicate edges BEFORE filtering or deduplicating."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    first = {}
    for i, row in enumerate(rows):
        keys = [('template', row['template_id']), ('dialogue', conversation_key(row))]
        keys += [('user', key) for key in sorted(user_keys(row))]
        for key in keys:
            if key in first:
                a, b = find(i), find(first[key])
                parent[max(a, b)] = min(a, b)
            else:
                first[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row['id'])
    if len(groups) < 3:
        raise ValueError('at least three independent prompt/template groups required')
    # Keep each task represented. Never split a connected component to meet ratios.
    buckets = {}
    for root in groups:
        tasks = tuple(sorted({row['task'] for i, row in enumerate(rows) if find(i) == root}))
        buckets.setdefault(tasks, []).append(root)
    owners = {}
    rng = random.Random(seed)
    for tasks, roots in sorted(buckets.items()):
        if len(roots) < 3:
            raise ValueError(f'at least three independent groups per task combination required: {tasks}')
        roots = sorted(roots)
        rng.shuffle(roots)
        heldout = max(1, round(len(roots)*.2))
        owners.update({root: 'valid' if n < heldout else 'test' if n < 2*heldout else 'train'
                       for n, root in enumerate(roots)})
    return {row['id']: (digest(encode(groups[find(i)])), owners[find(i)])
            for i, row in enumerate(rows)}


def stats(rows):
    return {'conversations': len(rows), 'tasks': dict(sorted(Counter(r['task'] for r in rows).items())),
            'groups': len({r['group_id'] for r in rows}),
            'assistant_turns': sum(m['role'] == 'assistant' for r in rows for m in r['messages']),
            'multiturn': sum(sum(m['role'] == 'user' for m in r['messages']) > 1 for r in rows),
            'content_bytes': sum(len(m['content'].encode('utf-8')) for r in rows for m in r['messages'])}


def build(input_path, sources_path, reviews_path, output, *, seed=1234, mode='draft'):
    if mode not in ('draft', 'reviewed'):
        raise ValueError('mode must be draft or reviewed')
    rows = read_jsonl(input_path)
    sources = json.loads(Path(sources_path).read_bytes())
    validate_records(rows, sources)
    rows = sorted(rows, key=lambda r: r['id'])
    receipts = review_index(rows, read_jsonl(reviews_path))
    assignments = assign_groups(rows, seed)
    splits = {name: [] for name in SPLITS}
    removed, excluded, seen = [], [], {}
    for row in rows:
        review = receipts.get(row['id'])
        if ((review and review['decision'] == 'rejected')
                or (mode == 'reviewed' and not review)):
            excluded.append(row['id'])
            continue
        key = conversation_key(row)
        if key in seen:
            removed.append({'id': row['id'], 'kept_id': seen[key]})
            continue
        seen[key] = row['id']
        group, split = assignments[row['id']]
        splits[split].append({'record': row, 'content_sha256': content_hash(row),
                              'group_id': group, 'review': review})
    if any(not splits[name] for name in SPLITS):
        raise ValueError('each split needs eligible records; complete human reviews or use draft mode')
    tasks = sorted({r['task'] for r in rows})
    if any(sorted({e['record']['task'] for e in entries}) != tasks for entries in splits.values()):
        raise ValueError('each split must cover all input tasks; review additional groups')
    raw_splits = {name: jsonl(records) for name, records in splits.items()}
    manifest = {'format': FORMAT, 'mode': mode, 'training_ready': mode == 'reviewed',
                'seed': seed, 'sources': sources, 'tasks': tasks,
                'inputs': {name: digest(Path(path).read_bytes()) for name, path in
                           [('examples', input_path), ('sources', sources_path), ('reviews', reviews_path)]},
                'split_policy': 'seeded task-combination-stratified connected components before review/dedup; every user turn + template + normalized dialogue; target 60/20/20 groups with at least one per split/task combination',
                'limitations': 'NFC/whitespace keys only; no semantic near-duplicate detection or independent benchmark claim. Review receipts record declarations, not authenticated identities.',
                'removed_duplicates': removed, 'excluded_ids': excluded, 'splits': {}}
    for name, entries in splits.items():
        records = [dict(e['record'], group_id=e['group_id']) for e in entries]
        manifest['splits'][name] = {'sha256': digest(raw_splits[name]), **stats(records)}
    output = Path(output).resolve()
    targets = {output/f'{name}.jsonl' for name in SPLITS} | {output/'manifest.json'}
    if targets & {Path(p).resolve() for p in (input_path, sources_path, reviews_path)}:
        raise ValueError('output must not overwrite source inputs')
    # Validate all content before creating or overwriting any export files.
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in raw_splits.items():
        (output/f'{name}.jsonl').write_bytes(raw)
    (output/'manifest.json').write_bytes(encode(manifest))
    load_bundle(output, for_training=mode == 'reviewed')
    return manifest


def load_bundle(directory, *, for_training=True):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_bytes())
    if manifest.get('format') != FORMAT or manifest.get('mode') not in ('draft', 'reviewed'):
        raise ValueError('unsupported SFT dataset format or mode')
    reviewed = manifest['mode'] == 'reviewed'
    if manifest.get('training_ready') is not reviewed or (for_training and not reviewed):
        raise ValueError('draft data is not human-reviewed training data')
    splits, owners, ids, dialogues = {}, {}, set(), set()
    for name in SPLITS:
        path = directory/f'{name}.jsonl'
        if digest(path.read_bytes()) != manifest['splits'][name]['sha256']:
            raise ValueError(f'{name}: file hash mismatch')
        entries = read_jsonl(path)
        if not entries:
            raise ValueError('empty split')
        records = [e['record'] for e in entries]
        validate_records(records, manifest['sources'])
        if sorted({r['task'] for r in records}) != manifest['tasks']:
            raise ValueError('split task coverage mismatch')
        for entry in entries:
            row = entry['record']; key = conversation_key(row)
            if (entry['content_sha256'] != content_hash(row) or row['id'] in ids or key in dialogues):
                raise ValueError('duplicate or corrupted conversation')
            ids.add(row['id']); dialogues.add(key)
            review = entry['review']
            if reviewed and not review:
                raise ValueError('reviewed split missing human receipt')
            if review:
                review_index([row], [review])
                if review['decision'] != 'approved':
                    raise ValueError('rejected record in export')
            group = entry['group_id']
            if not isinstance(group, str) or not group:
                raise ValueError('group id required')
            keys = [('group', group), ('template', row['template_id'])]
            keys += [('user', k) for k in user_keys(row)]
            for field in keys:
                if field in owners and owners[field] != name:
                    raise ValueError('prompt/template split leakage')
                owners[field] = name
        actual_stats = stats([dict(e['record'], group_id=e['group_id']) for e in entries])
        if any(manifest['splits'][name].get(k) != v for k, v in actual_stats.items()):
            raise ValueError('split statistics mismatch')
        splits[name] = records
    return manifest, splits


def legacy_audit(directory):
    """Inspect old marker text; never silently convert it into approved gold."""
    result, keys = {}, {}
    for name in ('train', 'valid'):
        raw = (Path(directory)/f'{name}.txt').read_bytes()
        rows, malformed, user_prompts, samples = [], [], set(), []
        for number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
            parts = re.split(r'(<사용자>|<봇>)', line)
            roles = parts[1::2]; contents = parts[2::2]
            if (parts[0].strip() or not roles or len(roles) % 2
                    or any(role != ('<사용자>' if i % 2 == 0 else '<봇>') for i, role in enumerate(roles))
                    or any(not content.strip() for content in contents)):
                malformed.append(number)
                continue
            rows.append(tuple(normalized(c) for c in contents))
            user_prompts.update(normalized(c) for c in contents[::2])
            if len(samples) < 5 and any(x in line for x in ('어디 있어?', '저는 자주 먹어요', '몸에 좋아요')):
                samples.append({'line': number, 'text': line})
        keys[name] = user_prompts
        result[name] = {'sha256': digest(raw), 'records': len(rows), 'malformed_lines': malformed,
                        'duplicate_dialogues': len(rows)-len(set(rows)),
                        'manual_review_samples': samples}
    result['shared_user_turns'] = len(keys['train'] & keys['valid'])
    result['decision'] = 'Legacy data kept unchanged; not imported as gold. No review/provenance receipts, template separation not established. Samples are review flags, not automatic error verdicts.'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE/'examples.jsonl')
    parser.add_argument('--sources', type=Path, default=HERE/'sources.json')
    parser.add_argument('--reviews', type=Path, default=HERE/'reviews.jsonl')
    parser.add_argument('--output-dir', type=Path, default=HERE/'draft')
    parser.add_argument('--mode', choices=['draft', 'reviewed'], default='draft')
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args(argv)
    report = build(args.input, args.sources, args.reviews, args.output_dir, seed=args.seed, mode=args.mode)
    print(json.dumps({'mode': report['mode'], 'training_ready': report['training_ready'],
                      'splits': report['splits']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
