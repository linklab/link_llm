"""Pack independent document chunks; score each text token and EOS exactly once."""
import torch
from torch.utils.data import DataLoader

POLICY = 'document-chunks-isolated-attention-reset-positions-v1'


def make_packs(lm, documents):
    if lm.BLOCK_SIZE < 1:
        raise ValueError('block_size must be positive')
    packs, current = [], {k: [] for k in ('x', 'y', 'segments', 'positions')}
    segment = 0
    for text in documents:
        ids = [lm.stoi[lm.BOS]] + lm.bpe.encode(text) + [lm.stoi[lm.END]]
        inputs, targets = ids[:-1], ids[1:]
        # A continuation chunk begins with the preceding text token, not a new BOS.
        # No target is dropped, repeated, or invented at a chunk boundary.
        for start in range(0, len(inputs), lm.BLOCK_SIZE):
            x = inputs[start:start + lm.BLOCK_SIZE]
            y = targets[start:start + lm.BLOCK_SIZE]
            if len(current['x']) + len(x) > lm.BLOCK_SIZE:
                packs.append(current)
                current = {k: [] for k in current}
                segment = 0
            current['x'].extend(x)
            current['y'].extend(y)
            current['segments'].extend([segment] * len(x))
            current['positions'].extend(range(len(x)))
            segment += 1
    if current['x']:
        packs.append(current)
    return packs


def collate(packs):
    length = max(len(p['x']) for p in packs)
    batch = {}
    for name, pad in [('x', 0), ('y', -100), ('segments', -1), ('positions', 0)]:
        value = torch.full((len(packs), length), pad, dtype=torch.long)
        for i, pack in enumerate(packs):
            value[i, :len(pack[name])] = torch.tensor(pack[name])
        batch[name] = value
    return batch


def batches(packs, batch_size, shuffle=False, seed=1234):
    return DataLoader(packs, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, generator=torch.Generator().manual_seed(seed))


def statistics(lm, documents, packs):
    lengths = [len(lm.bpe.encode(text)) + 1 for text in documents]
    sliding_rows = sum(1 + max(0, n - lm.BLOCK_SIZE) for n in lengths)
    sliding_inputs = sum(min(n, lm.BLOCK_SIZE) + max(0, n-lm.BLOCK_SIZE)*lm.BLOCK_SIZE
                         for n in lengths)
    targets = sum(lengths)
    return {'policy': POLICY, 'documents': len(documents), 'targets': targets,
            'packs': len(packs), 'capacity_tokens': len(packs)*lm.BLOCK_SIZE,
            'capacity_utilization': targets/(len(packs)*lm.BLOCK_SIZE) if packs else 0,
            'packed_input_tokens': sum(len(p['x']) for p in packs),
            'sliding_rows': sliding_rows, 'sliding_input_tokens': sliding_inputs,
            'long_documents': sum(n > lm.BLOCK_SIZE for n in lengths),
            'cross_document_attention': False, 'cross_chunk_attention': False,
            'position_policy': 'reset to zero at each document chunk',
            'evaluation_context': 'unchanged v0.4.0 sliding window; training context differs'}
