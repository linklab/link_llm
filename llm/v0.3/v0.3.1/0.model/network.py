"""v0.3.1: 여러 attention head를 연결하고 출력 projection으로 섞습니다."""
import math

import torch
from torch import nn


class MultiHeadLM(nn.Module):
    def __init__(self, vocab_size, embed, block_size, heads):
        super().__init__()
        if not isinstance(heads, int) or heads < 1 or embed < 1 or embed % heads:
            raise ValueError("heads는 양수 정수이며 embed는 heads로 나누어져야 합니다.")
        if block_size < 1:
            raise ValueError("block_size는 양수여야 합니다.")
        self.block_size, self.heads, self.head_dim = block_size, heads, embed // heads
        self.embedding = nn.Embedding(vocab_size, embed, padding_idx=0)
        # 하나의 큰 projection을 계산한 뒤 출력 차원을 head별로 나눕니다.
        self.query = nn.Linear(embed, embed, bias=False)
        self.key = nn.Linear(embed, embed, bias=False)
        self.value = nn.Linear(embed, embed, bias=False)
        self.projection = nn.Linear(embed, embed, bias=False)
        self.head = nn.Linear(embed, vocab_size)
        self.register_buffer("causal", torch.ones(block_size, block_size,
                                                  dtype=torch.bool).tril(), persistent=False)

    def embed_tokens(self, tokens):
        return self.embedding(tokens)

    def attention(self, h, tokens):
        batch, length = tokens.shape

        def split(layer):
            return layer(h).reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.query), split(self.key), split(self.value)  # (B,H,T,D)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        allowed = (self.causal[:length, :length][None, None]
                   & tokens.ne(0)[:, None, None, :])
        scores = scores.masked_fill(~allowed, float("-inf"))
        scores = scores.masked_fill(~allowed.any(dim=-1, keepdim=True), 0.0)
        weights = scores.softmax(dim=-1).masked_fill(~allowed, 0.0)
        merged = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, -1)
        return self.projection(merged) * tokens.ne(0).unsqueeze(-1)

    def transform(self, h, tokens):
        return self.attention(h, tokens)

    def forward(self, tokens):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.block_size:
            raise ValueError("입력은 (B,T), 1 <= T <= block_size 이어야 합니다.")
        hidden = self.transform(self.embed_tokens(tokens), tokens)
        logits = self.head(hidden)
        return logits.masked_fill(
            torch.arange(logits.shape[-1], device=tokens.device).eq(0), -1e9)
