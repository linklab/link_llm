"""외부 라이브러리 없는 결정적 UTF-8 바이트 BPE. 일반 문자열의 특수 토큰 해석 금지."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

SPECIALS = ("<PAD>", "<EOS>", "<BOS>", "<USER>", "<ASSISTANT>", "<SYSTEM>")
OFFSET = len(SPECIALS)
PATTERN = r"\s+|[^\s]+"


def merge(sequence, pair, new_id):
    result = []
    i = 0
    while i < len(sequence):
        if i + 1 < len(sequence) and (sequence[i], sequence[i+1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(sequence[i])
            i += 1
    return tuple(result)


def corpus_hash(documents):
    digest = hashlib.sha256()
    for text in documents:
        raw = text.encode('utf-8')
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)
    return digest.hexdigest()


class ByteBPE:
    def __init__(self):
        self.pieces = [bytes([i]) for i in range(256)]
        self.merges = []
        self.training_hash = None
        self.requested_vocab_size = OFFSET + 256
        self._refresh()

    def _refresh(self):
        self._cache = {}
        self.ranks = {tuple(pair): (rank, OFFSET+256+rank) for rank, pair in enumerate(self.merges)}
        self.tokens = list(SPECIALS) + ['b:' + piece.hex() for piece in self.pieces]

    @classmethod
    def train(cls, documents, vocab_size=768):
        if type(vocab_size) is not int or vocab_size < OFFSET+256:
            raise ValueError('vocab_size는 특수 토큰 6개 + 바이트 256개 이상이어야 합니다.')
        documents = list(documents)
        if not documents:
            raise ValueError('토크나이저 학습 문서가 없습니다.')
        result = cls()
        result.training_hash = corpus_hash(documents)
        result.requested_vocab_size = vocab_size
        words = Counter(tuple(b+OFFSET for b in chunk.encode('utf-8'))
                        for text in documents for chunk in re.findall(PATTERN, text))
        existing = set(result.pieces)
        while len(result.tokens) < vocab_size:
            pairs = Counter()
            for word, count in words.items():
                for pair in zip(word, word[1:]):
                    pairs[pair] += count
            candidates = [(pair, count) for pair, count in pairs.items()
                          if result.pieces[pair[0]-OFFSET] + result.pieces[pair[1]-OFFSET] not in existing]
            if not candidates:
                break
            # 빈도 동률이면 ID 사전순. 데이터의 해시/입력 순서 외 무작위성 없음.
            pair, _ = min(candidates, key=lambda item: (-item[1], item[0]))
            piece = result.pieces[pair[0]-OFFSET] + result.pieces[pair[1]-OFFSET]
            new_id = OFFSET + len(result.pieces)
            result.pieces.append(piece)
            existing.add(piece)
            result.merges.append(list(pair))
            updated = Counter()
            for word, count in words.items():
                updated[merge(word, pair, new_id)] += count
            words = updated
            result._refresh()
        return result

    def encode(self, text):
        ids = []
        for chunk in re.findall(PATTERN, text):
            cached = self._cache.get(chunk)
            if cached is not None:
                ids.extend(cached)
                continue
            sequence = tuple(b+OFFSET for b in chunk.encode('utf-8'))
            while len(sequence) > 1:
                options = [(self.ranks[pair][0], pair, self.ranks[pair][1])
                           for pair in zip(sequence, sequence[1:]) if pair in self.ranks]
                if not options:
                    break
                _, pair, new_id = min(options)
                sequence = merge(sequence, pair, new_id)
            if len(self._cache) >= 32768:
                self._cache.clear()
            self._cache[chunk] = sequence
            ids.extend(sequence)
        return ids

    def special_id(self, name):
        if name not in SPECIALS:
            raise ValueError('등록되지 않은 특수 토큰')
        return SPECIALS.index(name)

    def decode(self, ids, skip_special=False, errors='strict'):
        raw = bytearray()
        for idx in ids:
            if type(idx) is not int or not 0 <= idx < len(self.tokens):
                raise ValueError('잘못된 토큰 ID')
            if idx < OFFSET:
                if not skip_special:
                    raw.extend(SPECIALS[idx].encode('utf-8'))
            else:
                raw.extend(self.pieces[idx-OFFSET])
        return raw.decode('utf-8', errors=errors)

    def to_dict(self):
        return {'format_version': 1, 'kind': 'byte-bpe-v1', 'pretokenization': PATTERN,
                'specials': list(SPECIALS), 'merges': self.merges,
                'training_hash': self.training_hash, 'requested_vocab_size': self.requested_vocab_size}

    @classmethod
    def from_dict(cls, data):
        if (data.get('format_version') != 1 or data.get('kind') != 'byte-bpe-v1'
                or data.get('pretokenization') != PATTERN or data.get('specials') != list(SPECIALS)):
            raise ValueError('토크나이저 형식/특수 토큰 불일치')
        result = cls()
        seen = set(result.pieces)
        for pair in data['merges']:
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(type(i) is not int or not OFFSET <= i < OFFSET+len(result.pieces) for i in pair)):
                raise ValueError('BPE 병합 ID가 유효하지 않습니다.')
            piece = result.pieces[pair[0]-OFFSET] + result.pieces[pair[1]-OFFSET]
            if piece in seen:
                raise ValueError('중복 BPE 조각')
            result.pieces.append(piece);seen.add(piece);result.merges.append(pair.copy())
        result.training_hash = data['training_hash']
        result.requested_vocab_size = data['requested_vocab_size']
        if type(result.requested_vocab_size) is not int or result.requested_vocab_size < OFFSET+len(result.pieces):
            raise ValueError('BPE 어휘 크기 불일치')
        result._refresh()
        return result

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
