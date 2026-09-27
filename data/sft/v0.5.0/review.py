"""Record a human's explicit review after inspecting the conversation and criteria.

This command records declarations, not authenticated identities. Do not invoke it
to label automatic checks or AI review as human approval.
"""
import argparse
import json
from pathlib import Path
import dataset as d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=d.HERE/'examples.jsonl')
    parser.add_argument('--reviews', type=Path, default=d.HERE/'reviews.jsonl')
    parser.add_argument('--id', required=True)
    parser.add_argument('--decision', choices=['approved', 'rejected'], required=True)
    parser.add_argument('--reviewer', required=True)
    parser.add_argument('--note', required=True)
    parser.add_argument('--checked', choices=d.CRITERIA, nargs='*', default=[])
    args = parser.parse_args()
    if args.input.resolve() == args.reviews.resolve():
        parser.error('reviews must not overwrite source examples')
    rows = d.read_jsonl(args.input)
    row = next((r for r in rows if r['id'] == args.id), None)
    if row is None:
        parser.error('unknown conversation id')
    receipt = {'id': args.id, 'content_sha256': d.content_hash(row),
               'decision': args.decision, 'reviewer_kind': 'human',
               'reviewer': args.reviewer, 'note': args.note,
               'checks': {key: key in args.checked for key in d.CRITERIA}}
    d.review_index([row], [receipt])
    existing = d.read_jsonl(args.reviews) if args.reviews.exists() else []
    updated = [r for r in existing if r['id'] != args.id]+[receipt]
    d.review_index(rows, updated)
    args.reviews.write_bytes(d.jsonl(sorted(updated, key=lambda r: r['id'])))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
