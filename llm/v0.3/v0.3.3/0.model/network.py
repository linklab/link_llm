"""단일 Pre-LN Transformer 블록. 다층 적층은 v0.3.4의 범위입니다."""
import importlib.util
from pathlib import Path

from torch import nn

spec = importlib.util.spec_from_file_location(
    "v033_position", Path(__file__).resolve().parents[2] / "v0.3.2/0.model/network.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


class TransformerLM(previous.PositionLM):
    def __init__(self, vocab_size, embed, block_size, heads, ffn_hidden=256, dropout=0.1):
        super().__init__(vocab_size, embed, block_size, heads)
        if not isinstance(ffn_hidden, int) or ffn_hidden < 1:
            raise ValueError("ffn_hidden은 양수 정수여야 합니다.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout은 0 이상 1 미만이어야 합니다.")
        self.ln1 = nn.LayerNorm(embed)
        self.ln2 = nn.LayerNorm(embed)
        self.ffn = nn.Sequential(nn.Linear(embed, ffn_hidden), nn.GELU(),
                                 nn.Linear(ffn_hidden, embed))
        self.dropout = nn.Dropout(dropout)

    def transform(self, h, tokens):
        mask = tokens.ne(0).unsqueeze(-1)
        h = (h + self.dropout(self.attention(self.ln1(h), tokens))) * mask
        return (h + self.dropout(self.ffn(self.ln2(h)))) * mask
