from typing import Optional, Tuple
from math import sqrt

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, Tensor

from .util import RMSNorm, SwiGLU


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super(Attention, self).__init__()
        assert d_model % n_heads == 0, "Transformer dimensions must be divisible by number of heads."

        self.scale = sqrt(d_model / n_heads)
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        q = rearrange(self.W_q(x), "b l (n d) -> b n l d", n=self.n_heads)
        k = rearrange(self.W_k(x), "b l (n d) -> b n l d", n=self.n_heads)
        v = rearrange(self.W_v(x), "b l (n d) -> b n l d", n=self.n_heads)

        return self.W_o(
            rearrange(
                self.dropout(F.softmax(
                    (q @ k.transpose(-2, -1)) / self.scale, dim=-1
                )) @ v,
                "b n l d -> b l (n d)"
            )
        )


class TransformerBlock(nn.Module):
    def __init__(
            self, d_model: int, n_heads: int, attn_dropout: float = 0.0, ffn_dropout: float = 0.0,
            d_hidden: Optional[int] = None, norm_eps: float = 1e-6
    ):
        super(TransformerBlock, self).__init__()
        self.attn = nn.Sequential(
            RMSNorm(d_model, eps=norm_eps),
            Attention(d_model, n_heads, attn_dropout)
        )

        self.ffn = nn.Sequential(
            RMSNorm(d_model, eps=norm_eps),
            SwiGLU(d_model, d_hidden),
            nn.Dropout(ffn_dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)

        return x + self.ffn(x)


class ViT(nn.Module):
    def __init__(
            self, in_channels: int, d_model: int, n_heads: int, n_layers: int,
            attn_dropout: float = 0.0, ffn_dropout: float = 0.0,
            n_cols: int = 7, n_rows: int = 6, patch_size: int = 2
    ):
        super(ViT, self).__init__()
        self.patch_emb = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

        self.cls_emb = nn.Parameter(
            torch.empty(1, 1, d_model).normal_(std=sqrt(1 / d_model))
        )
        self.pos_emb = nn.Parameter(
            torch.empty(1, (n_cols // patch_size) * (n_rows // patch_size), d_model).normal_(std=sqrt(1 / d_model))
        )
        self.transformer = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, attn_dropout, ffn_dropout)
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, 7)
        )

    def forward(self, grid: Tensor) -> Tensor:
        B = grid.shape[0]

        x = torch.concatenate((
            self.cls_emb.expand(B, -1, -1),
            self.pos_emb + rearrange(
                self.patch_emb(grid), "b c h w -> b (h w) c"
            )
        ), dim=1)

        x = self.transformer(x)

        return self.head(x[:, 0])
