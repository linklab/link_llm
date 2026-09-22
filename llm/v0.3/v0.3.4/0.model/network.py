"""N층 Pre-LN decoder와 GPT-2식 residual projection 초기화."""
import importlib.util
import math
from pathlib import Path

import torch
from torch import nn

spec = importlib.util.spec_from_file_location(
    "v034_attention", Path(__file__).resolve().parents[2] / "v0.3.1/0.model/network.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


class Block(nn.Module):
    # 검증된 head 분할·인과/PAD 마스크 계산을 재사용합니다.
    attention = previous.MultiHeadLM.attention

    def __init__(self, embed, block_size, heads, ffn_hidden, dropout):
        super().__init__()
        self.heads, self.head_dim = heads, embed // heads
        self.query = nn.Linear(embed, embed, bias=False)
        self.key = nn.Linear(embed, embed, bias=False)
        self.value = nn.Linear(embed, embed, bias=False)
        self.projection = nn.Linear(embed, embed, bias=False)
        self.register_buffer("causal", torch.ones(block_size, block_size,
                             dtype=torch.bool).tril(), persistent=False)
        self.ln1, self.ln2 = nn.LayerNorm(embed), nn.LayerNorm(embed)
        self.ffn = nn.Sequential(nn.Linear(embed, ffn_hidden), nn.GELU(),
                                 nn.Linear(ffn_hidden, embed))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, tokens):
        mask = tokens.ne(0).unsqueeze(-1)
        h = (h + self.dropout(self.attention(self.ln1(h), tokens))) * mask
        return (h + self.dropout(self.ffn(self.ln2(h)))) * mask


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed, block_size, heads, ffn_hidden=256,
                 dropout=0.1, layers=4):
        super().__init__()
        for name, value in (("vocab_size", vocab_size), ("embed", embed),
                            ("block_size", block_size), ("heads", heads),
                            ("ffn_hidden", ffn_hidden), ("layers", layers)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name}은 양수 정수여야 합니다.")
        if embed % heads:
            raise ValueError("embed는 heads로 나누어져야 합니다.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout은 0 이상 1 미만이어야 합니다.")
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embed, padding_idx=0)
        self.position = nn.Embedding(block_size, embed)
        self.blocks = nn.ModuleList([Block(embed, block_size, heads, ffn_hidden, dropout)
                                     for _ in range(layers)])
        self.final_norm = nn.LayerNorm(embed)
        self.head = nn.Linear(embed, vocab_size)
        self.apply(self._init_weights)
        # 블록마다 residual branch가 두 개이므로 깊이에 따라 분산을 줄입니다.
        residual_std = 0.02 / math.sqrt(2 * layers)
        for block in self.blocks:
            nn.init.normal_(block.projection.weight, std=residual_std)
            nn.init.normal_(block.ffn[2].weight, std=residual_std)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def embed_tokens(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return (self.embedding(tokens) + self.position(positions)) * tokens.ne(0).unsqueeze(-1)

    def forward(self, tokens):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.block_size:
            raise ValueError("입력은 (B,T), 1 <= T <= block_size 이어야 합니다.")
        h = self.embed_tokens(tokens)
        for block in self.blocks:
            h = block(h, tokens)
        logits = self.head(self.final_norm(h) * tokens.ne(0).unsqueeze(-1))
        return logits.masked_fill(torch.arange(logits.shape[-1], device=tokens.device).eq(0), -1e9)
