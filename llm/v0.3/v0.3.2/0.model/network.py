"""학습형 위치 임베딩. 위치는 각 슬라이딩 문맥창에서 0부터 시작합니다."""
import importlib.util
from pathlib import Path

import torch
from torch import nn

spec = importlib.util.spec_from_file_location(
    "v032_attention", Path(__file__).resolve().parents[2] / "v0.3.1/0.model/network.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


class PositionLM(previous.MultiHeadLM):
    def __init__(self, vocab_size, embed, block_size, heads):
        super().__init__(vocab_size, embed, block_size, heads)
        self.position = nn.Embedding(block_size, embed)

    def embed_tokens(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return (self.embedding(tokens) + self.position(positions)) * tokens.ne(0).unsqueeze(-1)
