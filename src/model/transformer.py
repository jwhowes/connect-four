from dataclasses import dataclass
from math import sqrt
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, FloatTensor

from . import BaseModel
from .. import NUM_ROWS, NUM_COLS
from ..util import Config


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(d_model).normal_(mean=1.0, std=sqrt(1 / d_model)))

    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: Optional[int] = None):
        super(SwiGLU, self).__init__()
        if d_hidden is None:
            d_hidden = 4 * d_model

        self.gate = nn.Linear(d_model, d_hidden, bias=False)
        self.hidden = nn.Linear(d_model, d_hidden, bias=False)
        self.out = nn.Linear(d_hidden, d_model)

    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.out(
            F.silu(self.gate(x)) * self.hidden(x)
        )


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super(Attention, self).__init__()
        assert d_model % n_heads == 0

        self.scale = sqrt(d_model // n_heads)
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: FloatTensor) -> FloatTensor:
        q = rearrange(self.W_q(x), "b l (n d) -> b n l d", n=self.n_heads)
        k = rearrange(self.W_k(x), "b l (n d) -> b n l d", n=self.n_heads)
        v = rearrange(self.W_v(x), "b l (n d) -> b n l d", n=self.n_heads)

        return self.W_o(
            rearrange(
                F.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1) @ v, "b n l d -> b l (n d)"
            )
        )


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_hidden: Optional[int] = None, norm_eps: float = 1e-6):
        super(Block, self).__init__()
        self.attn = Attention(d_model, n_heads)
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)

        self.ffn = SwiGLU(d_model, d_hidden)
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = x + self.attn(self.attn_norm(x))

        return x + self.ffn(self.ffn_norm(x))


class Transformer(BaseModel):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, norm_eps: float = 1e-6):
        super(Transformer, self).__init__()
        self.emb = nn.Linear(3, d_model)

        self.pos_emb = nn.Parameter(
            torch.empty(1, NUM_ROWS, NUM_COLS, d_model).normal_(std=sqrt(1 / d_model))
        )
        self.winner_emb = nn.Parameter(
            torch.empty(1, 1, d_model).normal_(std=sqrt(1 / d_model))
        )
        self.action_emb = nn.Parameter(
            torch.empty(1, 1, d_model).normal_(std=sqrt(1 / d_model))
        )

        self.layers = nn.Sequential(*[
            Block(d_model, n_heads, norm_eps=norm_eps) for _ in range(n_layers)
        ])

        self.winner_head = nn.Sequential(
            RMSNorm(d_model, eps=norm_eps),
            nn.Linear(d_model, 3)
        )

        self.action_head = nn.Sequential(
            RMSNorm(d_model, eps=norm_eps),
            nn.Linear(d_model, NUM_COLS)
        )

    def forward(self, board: FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
        B = board.shape[0]
        x = torch.concatenate((
            self.winner_emb.expand(B, -1, -1),
            self.action_emb.expand(B, -1, -1),
            (self.pos_emb + self.emb(board)).flatten(1, 2)
        ), dim=1)

        x = self.layers(x)

        return self.winner_head(x[:, 0]), self.action_head(x[:, 1])


@dataclass
class TransformerConfig[T: Transformer](Config):
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    norm_eps: float = 1e-6
