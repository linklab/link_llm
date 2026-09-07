"""문장 경계를 보존하는 시퀀스 창과 오른쪽 padding.

첫 창에서는 모든 다음 토큰을 학습하고, 문맥창을 넘은 뒤에는 슬라이딩 창의
마지막 정답만 학습합니다. 모든 정답은 정확히 한 번 등장하며, 각 정답의
문맥은 generate()의 최근 block_size 토큰과 같습니다. 대규모 코퍼스의
패킹/효율화는 v0.4에서 다룹니다.
"""
import torch
from torch.utils.data import DataLoader


def make_windows(lm, sentences):
    rows = []
    for sentence in sentences:
        tokens = lm.prepare(lm.tokenize(sentence))
        ids = [lm.stoi.get(t, 0) for t in tokens]
        if len(ids) < 2:
            continue
        end = min(lm.BLOCK_SIZE, len(ids) - 1)
        x, y = ids[:end], ids[1:end + 1]
        known = [all(ids[:i + 1]) and ids[i + 1] != 0 for i in range(end)]
        rows.append((x, y, known))
        for i in range(end + 1, len(ids)):
            x = ids[i - lm.BLOCK_SIZE:i]
            y = [-100] * (len(x) - 1) + [ids[i]]
            known = [False] * (len(x) - 1) + [all(x) and ids[i] != 0]
            rows.append((x, y, known))
    return rows


def collate(rows):
    length = max(len(row[0]) for row in rows)
    x = torch.zeros((len(rows), length), dtype=torch.long)
    y = torch.full_like(x, -100)
    known = torch.zeros_like(x, dtype=torch.bool)
    for i, (inputs, targets, valid) in enumerate(rows):
        n = len(inputs)
        x[i, :n] = torch.tensor(inputs)
        y[i, :n] = torch.tensor(targets)
        known[i, :n] = torch.tensor(valid)
    return x, y, known


def batches(rows, batch_size, shuffle=False, generator=None):
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, generator=generator)
