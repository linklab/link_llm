"""Build or verify the complete v0.5.0–v0.5.1 preparation handoff."""
import argparse
import json
from pathlib import Path
import tempfile

from prepare import module, prepare, load_prepared, VERSION, ROOT, DEFAULT_TOKENIZER

data = module('release051_data', ROOT/'data/sft/v0.5.0/dataset.py')
audit = module('release051_verify', VERSION/'2.test/test.py')
setup = module('release051_setup', VERSION/'0.model/model_setup.py')
SOURCE = ROOT/'data/sft/v0.5.0'
DEFAULT_OUTPUT = VERSION/'1.train/release'


def validate_quality(rows, quality):
    if (quality.get('format') != 'sft-ai-content-audit-v1'
            or quality.get('reviewer_kind') != 'ai'
            or quality.get('human_approval') is not False
            or quality.get('criteria') != list(data.CRITERIA)):
        raise ValueError('AI quality audit identity/criteria mismatch')
    receipts = quality.get('records', [])
    if len(receipts) != len(rows) or {r['id'] for r in receipts} != {r['id'] for r in rows}:
        raise ValueError('quality audit must cover every record exactly once')
    by_id = {r['id']: r for r in rows}
    for receipt in receipts:
        row = by_id[receipt['id']]
        if (receipt.get('content_sha256') != data.content_hash(row)
                or receipt.get('decision') != 'suitable_for_educational_experiment'
                or receipt.get('assistant_responses_checked') != sum(
                    m['role'] == 'assistant' for m in row['messages'])
                or not isinstance(receipt.get('note'), str) or not receipt['note'].strip()):
            raise ValueError('stale or incomplete AI quality audit')


def verify(directory):
    directory = Path(directory)
    manifest, template, splits, _ = load_prepared(directory/'prepared', for_training=False)
    rows = data.read_jsonl(directory/'inputs/examples.jsonl')
    quality = json.loads((directory/'inputs/ai_quality_audit.json').read_bytes())
    validate_quality(rows, quality)
    # Rebuild from the archived inputs; validate split membership and provenance.
    with tempfile.TemporaryDirectory() as tmp:
        data.build(directory/'inputs/examples.jsonl', directory/'inputs/sources.json',
                   directory/'inputs/reviews.jsonl', tmp)
        for name in ['manifest.json', 'train.jsonl', 'valid.jsonl', 'test.jsonl']:
            if (Path(tmp)/name).read_bytes() != (directory/'prepared/source'/name).read_bytes():
                raise ValueError('release source differs from archived inputs')
    config = json.loads((directory/'model_config.json').read_bytes())
    if config != setup.configuration(template):
        raise ValueError('release model configuration mismatch')
    result = audit.verify(directory/'prepared', allow_draft=True)
    maximum = max(s['statistics']['max_tokens'] for s in manifest['splits'].values())
    if maximum > config['block_size']:
        raise ValueError('release exceeds configured model context; no truncation allowed')
    return {'format': 'sft-preparation-release-v1', 'preparation_complete': True,
            'training_ready': manifest['training_ready'], 'reviewer_kind': 'ai',
            'human_approval': False, 'model_weights_trained': False,
            'model_configuration_complete': True, 'block_size': config['block_size'],
            'max_conversation_tokens': maximum,
            'ai_reviewed_conversations': len(rows),
            'validation': result,
            'remaining': ['Human review for reviewed-data training gate',
                          'v0.5.2 response loss masking',
                          'v0.5.3 base-weight migration and SFT training'],
            'files': {str(p.relative_to(directory)): data.digest(p.read_bytes())
                      for p in sorted(directory.rglob('*'))
                      if p.is_file() and p != directory/'readiness.json'}}


def build(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    if output.is_relative_to(SOURCE) or SOURCE.is_relative_to(output):
        raise ValueError('release output must not overlap original data')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='release051-', dir=output.parent) as tmp:
        staged = Path(tmp)/'release'
        inputs = staged/'inputs'
        inputs.mkdir(parents=True)
        for name in ['examples.jsonl', 'sources.json', 'reviews.jsonl', 'ai_quality_audit.json']:
            (inputs/name).write_bytes((SOURCE/name).read_bytes())
        rows = data.read_jsonl(inputs/'examples.jsonl')
        validate_quality(rows, json.loads((inputs/'ai_quality_audit.json').read_bytes()))
        source = Path(tmp)/'source'
        data.build(inputs/'examples.jsonl', inputs/'sources.json', inputs/'reviews.jsonl', source)
        prepare(source, DEFAULT_TOKENIZER, staged/'prepared', allow_draft=True, max_tokens=256)
        _, template, _, _ = load_prepared(staged/'prepared', for_training=False)
        (staged/'model_config.json').write_bytes(data.encode(setup.configuration(template)))
        result = verify(staged)
        (staged/'readiness.json').write_bytes(data.encode(result))
        if output.exists():
            expected = {str(p.relative_to(staged)): p.read_bytes()
                        for p in staged.rglob('*') if p.is_file()}
            actual = {str(p.relative_to(output)): p.read_bytes()
                      for p in output.rglob('*') if p.is_file()} if output.is_dir() else {}
            if actual != expected:
                raise FileExistsError('release differs; choose a new --output-dir')
        else:
            staged.rename(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true', help='Verify an existing release without writing')
    args = parser.parse_args()
    if args.check:
        result = verify(args.output_dir)
        if json.loads((args.output_dir/'readiness.json').read_bytes()) != result:
            raise ValueError('readiness report differs from actual release')
    else:
        result = build(args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'validation')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
