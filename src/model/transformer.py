from dataclasses import dataclass
from math import sqrt
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, FloatTensor

from . import BaseModel
from .. import NUM_ROWS, NUM_COLS
from ..util import Config

class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super(Attention, self).__init__()
        assert d_model % n_heads == 0

        self.scale = sqrt(d_model // n_heads)
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: FloatTensor) -> FloatTensor:
        q = rearrange(self.W_q(x), "b l (n d) -> b n l d", n=self.n_heads)
        k = rearrange(self.W_k(x), "b l (n d) -> b n l d", n=self.n_heads)
        v = rearrange(self.W_v(x), "b l (n d) -> b n l d", n=self.n_heads)

        return self.W_o(
            rearrange(
                self.dropout(F.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1)) @ v, "b n l d -> b l (n d)"
            )
        )


class Block(nn.Module):
    def __init__(
            self, d_model: int, n_heads: int, d_hidden: Optional[int] = None, norm_eps: float = 1e-6,
            attn_dropout: float = 0.0, resid_dropout: float = 0.0
    ):
        super(Block, self).__init__()
        if d_hidden is None:
            d_hidden = 4 * d_model
        self.attn = Attention(d_model, n_heads, dropout=attn_dropout)
        self.attn_norm = nn.LayerNorm(d_model, eps=norm_eps)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.ffn_dropout = nn.Dropout(resid_dropout)

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = x + self.attn(self.attn_norm(x))

        return x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))


class Transformer(BaseModel):
    def __init__(
            self, d_model: int, n_layers: int, n_heads: int, norm_eps: float = 1e-6,
            attn_dropout: float = 0.0, resid_dropout: float = 0.0
    ):
        super(Transformer, self).__init__()
        self.emb = nn.Linear(3, d_model)

        self.col_emb = nn.Parameter(
            torch.empty(1, NUM_COLS, d_model).normal_(std=sqrt(1 / d_model))
        )
        self.row_emb = nn.Parameter(
            torch.empty(1, NUM_ROWS, d_model).normal_(std=sqrt(1 / d_model))
        )
        self.cls_emb = nn.Parameter(
            torch.empty(1, 1, d_model).normal_(std=sqrt(1 / d_model))
        )

        self.layers = nn.Sequential(*[
            Block(
                d_model, n_heads, norm_eps=norm_eps, attn_dropout=attn_dropout, resid_dropout=resid_dropout
            ) for _ in range(n_layers)
        ])

        self.head = nn.Linear(d_model, 3)

    def forward(self, board: FloatTensor) -> FloatTensor:
        B = board.shape[0]
        x = torch.concatenate((
            self.cls_emb.expand(B, -1, -1),
            (self.col_emb.unsqueeze(1) + self.row_emb.unsqueeze(2) + self.emb(board)).flatten(1, 2)
        ), dim=1)

        x = self.layers(x)

        return self.head(x[:, 0])


@dataclass
class TransformerConfig[T: Transformer](Config):
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    norm_eps: float = 1e-6
