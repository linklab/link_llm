"""Validate prepared transcripts and teacher-forced/inference prefix equality."""
import argparse
import importlib.util
import json
from pathlib import Path

VERSION = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare051_audit', VERSION/'1.train/prepare.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def verify(directory, *, allow_draft=False):
    manifest, template, splits, encoded = p.load_prepared(directory, for_training=not allow_draft)
    count, responses = 0, 0
    for name, records in splits.items():
        for row, entry in zip(records, encoded[name]):
            if template.decode_conversation(entry['input_ids']) != row['messages']:
                raise ValueError('raw message roundtrip failed')
            count += 1
            for i, span in enumerate(entry['message_spans']):
                if span['role'] != 'assistant':
                    continue
                prompt = template.encode_prompt(row['messages'][:i])
                if prompt['input_ids'] != entry['input_ids'][:span['content_start']]:
                    raise ValueError('training/inference prefix mismatch')
                response = template.decode_response(entry['input_ids'][span['content_start']:span['end']])
                if response['text'] != row['messages'][i]['content'] or response['stop_reason'] != 'turn_end':
                    raise ValueError('assistant response boundary mismatch')
                responses += 1
    return {'version': 'v0.5.1', 'template_sha256': template.fingerprint,
            'training_ready': manifest['training_ready'], 'conversation_roundtrips': count,
            'assistant_prefix_and_response_checks': responses,
            'base_vocab_size': template.base_vocab_size, 'chat_vocab_size': template.vocab_size,
            'eot_id': template.eot_id, 'splits': manifest['splits'],
            'note': 'Formatting validation only; no model generation quality or human review claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=VERSION/'1.train/preview')
    parser.add_argument('--allow-draft', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset, allow_draft=args.allow_draft), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
