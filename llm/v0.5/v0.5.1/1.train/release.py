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


def validate_approvals(rows, reviews):
    receipts = data.review_index(rows, reviews)
    if set(receipts) != {row['id'] for row in rows} or any(
            review['decision'] != 'approved' for review in receipts.values()):
        raise ValueError('complete human approval required for every release conversation')
    return receipts


def verify(directory):
    directory = Path(directory)
    manifest, template, splits, _ = load_prepared(directory/'prepared', for_training=True)
    rows = data.read_jsonl(directory/'inputs/examples.jsonl')
    receipts = validate_approvals(rows, data.read_jsonl(directory/'inputs/reviews.jsonl'))
    # Rebuild from the archived inputs; validate split membership and provenance.
    with tempfile.TemporaryDirectory() as tmp:
        data.build(directory/'inputs/examples.jsonl', directory/'inputs/sources.json',
                   directory/'inputs/reviews.jsonl', tmp, mode='reviewed')
        for name in ['manifest.json', 'train.jsonl', 'valid.jsonl', 'test.jsonl']:
            if (Path(tmp)/name).read_bytes() != (directory/'prepared/source'/name).read_bytes():
                raise ValueError('release source differs from archived inputs')
    config = json.loads((directory/'model_config.json').read_bytes())
    if config != setup.configuration(template):
        raise ValueError('release model configuration mismatch')
    result = audit.verify(directory/'prepared')
    maximum = max(s['statistics']['max_tokens'] for s in manifest['splits'].values())
    if maximum > config['block_size']:
        raise ValueError('release exceeds configured model context; no truncation allowed')
    return {'format': 'sft-preparation-release-v2', 'preparation_complete': True,
            'training_ready': manifest['training_ready'], 'reviewer_kind': 'human',
            'human_approval': True, 'model_weights_trained': False,
            'model_configuration_complete': True, 'block_size': config['block_size'],
            'max_conversation_tokens': maximum,
            'human_approved_conversations': len(receipts),
            'validation': result,
            'remaining_in_v0.5.1': [],
            'next_versions': ['v0.5.2 response loss masking',
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
        for name in ['examples.jsonl', 'sources.json', 'reviews.jsonl']:
            (inputs/name).write_bytes((SOURCE/name).read_bytes())
        rows = data.read_jsonl(inputs/'examples.jsonl')
        validate_approvals(rows, data.read_jsonl(inputs/'reviews.jsonl'))
        source = Path(tmp)/'source'
        data.build(inputs/'examples.jsonl', inputs/'sources.json', inputs/'reviews.jsonl', source, mode='reviewed')
        prepare(source, DEFAULT_TOKENIZER, staged/'prepared', max_tokens=256)
        _, template, _, _ = load_prepared(staged/'prepared', for_training=True)
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
