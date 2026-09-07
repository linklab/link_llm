"""v0.3.0: embedding → single causal attention head → vocabulary logits.

위치 임베딩, residual, LayerNorm, FFN은 아직 도입하지 않습니다.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SingleHeadLM(nn.Module):
    def __init__(self, vocab_size, embed, block_size):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embed, padding_idx=0)
        self.query = nn.Linear(embed, embed, bias=False)
        self.key = nn.Linear(embed, embed, bias=False)
        self.value = nn.Linear(embed, embed, bias=False)
        self.head = nn.Linear(embed, vocab_size)
        self.register_buffer("causal", torch.ones(block_size, block_size,
                                                  dtype=torch.bool).tril(), persistent=False)

    def forward(self, tokens):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.block_size:
            raise ValueError("입력은 (B,T), 1 <= T <= block_size 이어야 합니다.")
        h = self.embedding(tokens)                          # (B,T,E)
        q, k, v = self.query(h), self.key(h), self.value(h)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        length = tokens.shape[1]
        allowed = self.causal[:length, :length] & tokens.ne(0).unsqueeze(1)
        scores = scores.masked_fill(~allowed, float("-inf"))
        # 전부 PAD인 query도 softmax(-inf,...) → NaN이 되지 않도록 처리합니다.
        empty = ~allowed.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(empty, 0.0)
        weights = scores.softmax(dim=-1).masked_fill(~allowed, 0.0)
        hidden = (weights @ v) * tokens.ne(0).unsqueeze(-1)
        logits = self.head(hidden)                          # (B,T,V)
        # PAD는 배치용 자리 표시이며 다음 토큰의 후보가 아닙니다.
        return logits.masked_fill(
            torch.arange(logits.shape[-1], device=tokens.device).eq(0), -1e9)


def sequence_loss(logits, targets):
    """오른쪽 PAD와 이미 채점한 overlap은 -100; 실제 정답만 평균합니다."""
    active = targets.ne(-100)
    if not active.any():
        raise ValueError("학습할 정답 토큰이 없는 배치입니다.")
    return F.cross_entropy(logits[active], targets[active])
