"""Versioned, reversible chat framing over an unchanged byte-BPE vocabulary.

No torch, model training, truncation or loss masking. Append EOT at the end of
the base vocabulary; never shift any pretrained token ID.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('v051_bpe', ROOT/'llm/v0.4/v0.4.0/0.model/tokenizer.py')
tokenizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tokenizer)
ByteBPE = tokenizer.ByteBPE
FORMAT = 'link-chat-v1'
ROLES = {'system': 5, 'user': 3, 'assistant': 4}
BOS, EOS = 2, 1


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode('utf-8')


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_messages(messages, *, generation=False):
    if not isinstance(messages, list) or not messages:
        raise ValueError('nonempty message list required')
    expected = 'user'
    user_seen = False
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict) or set(msg) != {'role', 'content'}:
            raise ValueError('each message requires only role and content')
        if not isinstance(msg['content'], str):
            raise ValueError('message content must be a string')
        try:
            msg['content'].encode('utf-8')
        except UnicodeEncodeError as error:
            raise ValueError('message contains invalid Unicode') from error
        if index == 0 and msg['role'] == 'system':
            continue
        if msg['role'] != expected:
            raise ValueError('roles must alternate user/assistant after optional initial system')
        user_seen |= msg['role'] == 'user'
        expected = 'assistant' if expected == 'user' else 'user'
    end_role = 'user' if generation else 'assistant'
    if not user_seen or messages[-1]['role'] != end_role:
        raise ValueError(f'conversation must end with {end_role}')


class ChatTemplate:
    def __init__(self, bpe):
        # Detach from a caller-owned tokenizer; no new BPE fitting on SFT data.
        self.bpe = ByteBPE.from_dict(bpe.to_dict())
        self.base_vocab_size = len(self.bpe.tokens)
        self.eot_id = self.base_vocab_size
        self.vocab_size = self.base_vocab_size + 1

    @property
    def stop_token_ids(self):
        return (self.eot_id, EOS)

    def to_dict(self):
        bpe_config = json.loads(json.dumps(self.bpe.to_dict()))
        return {'format': FORMAT, 'tokenizer': bpe_config,
                'base_tokenizer_sha256': sha256(json_bytes(bpe_config)),
                'roles': dict(ROLES), 'bos_id': BOS, 'eos_id': EOS, 'eot_id': self.eot_id,
                'base_vocab_size': self.base_vocab_size, 'vocab_size': self.vocab_size,
                'policy': 'BOS + (role + raw byte-BPE content + EOT)* + EOS; generation opens ASSISTANT'}

    @property
    def fingerprint(self):
        return sha256(json_bytes(self.to_dict()))

    def save(self, path):
        Path(path).write_bytes(json_bytes(self.to_dict()))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_bytes())
        instance = cls(ByteBPE.from_dict(data['tokenizer']))
        if data != instance.to_dict():
            raise ValueError('chat template metadata/tokenizer mismatch')
        return instance

    def _encode(self, messages, generation):
        validate_messages(messages, generation=generation)
        ids, spans = [BOS], []
        for msg in messages:
            start = len(ids)
            ids.append(ROLES[msg['role']])
            content_start = len(ids)
            ids.extend(self.bpe.encode(msg['content']))
            content_end = len(ids)
            ids.append(self.eot_id)
            spans.append({'role': msg['role'], 'start': start,
                          'content_start': content_start, 'content_end': content_end,
                          'end': len(ids)})
        result = {'input_ids': ids, 'message_spans': spans,
                  'template_sha256': self.fingerprint}
        if generation:
            ids.append(ROLES['assistant'])
            result['response_start'] = len(ids)
        else:
            result['eos_index'] = len(ids)
            ids.append(EOS)
        return result

    def encode_conversation(self, messages):
        """Complete training transcript; positions are half-open, no loss mask."""
        return self._encode(messages, False)

    def encode_prompt(self, messages):
        """History through the latest user + open assistant header, without EOS."""
        return self._encode(messages, True)

    def _check_ids(self, ids):
        if not isinstance(ids, (list, tuple)) or any(
                type(i) is not int or not 0 <= i < self.vocab_size for i in ids):
            raise ValueError('invalid chat token IDs')

    def decode_conversation(self, ids):
        self._check_ids(ids)
        if len(ids) < 2 or ids[0] != BOS or ids[-1] != EOS:
            raise ValueError('complete conversation requires BOS and final EOS')
        inverse = {value: key for key, value in ROLES.items()}
        messages, cursor = [], 1
        while cursor < len(ids)-1:
            if ids[cursor] not in inverse:
                raise ValueError('message role token expected')
            role = inverse[ids[cursor]]
            cursor += 1
            start = cursor
            while cursor < len(ids)-1 and ids[cursor] != self.eot_id:
                if ids[cursor] < len(tokenizer.SPECIALS):
                    raise ValueError('structural token inside message content')
                cursor += 1
            if cursor >= len(ids)-1:
                raise ValueError('message EOT missing')
            text = self.bpe.decode(ids[start:cursor], errors='strict')
            messages.append({'role': role, 'content': text})
            cursor += 1
        validate_messages(messages)
        return messages

    def decode_response(self, ids):
        """Decode newly generated IDs only; reject role injection/trailing tokens.

        EOT ends an assistant turn. EOS ends the conversation. An EOT+EOS pair
        is accepted for a complete training suffix. No terminator => incomplete.
        Streaming partial UTF-8 handling belongs to a future streaming adapter.
        """
        self._check_ids(ids)
        end = len(ids)
        stop = 'incomplete'
        for index, token in enumerate(ids):
            if token in self.stop_token_ids:
                end = index
                suffix = list(ids[index:])
                if suffix == [self.eot_id]:
                    stop = 'turn_end'
                elif suffix in ([EOS], [self.eot_id, EOS]):
                    stop = 'conversation_end'
                else:
                    raise ValueError('unexpected tokens after response terminator')
                break
            if token < len(tokenizer.SPECIALS):
                raise ValueError('response cannot contain role/BOS/PAD tokens')
        return {'text': self.bpe.decode(ids[:end], errors='strict'),
                'stop_reason': stop, 'conversation_closed': stop == 'conversation_end'}
