from __future__ import annotations

from typing import Optional, Tuple

from torch import nn, FloatTensor

from .base import BaseModel
from .. import NUM_COLS


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: FloatTensor) -> FloatTensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, d_model: int, d_hidden: Optional[int] = None, norm_eps: float = 1e-6):
        super(Block, self).__init__()
        if d_hidden is None:
            d_hidden = 4 * d_model

        self.module = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=5, padding=2),
            LayerNorm2d(d_model, eps=norm_eps),
            nn.Conv2d(d_model, d_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_hidden, d_model, kernel_size=1)
        )

    def forward(self, x: FloatTensor) -> FloatTensor:
        return x + self.module(x)


class ConvModel(BaseModel):
    def __init__(
            self, dims: Tuple[int, ...], depths: Tuple[int, ...], norm_eps: float = 1e-6
    ):
        super(ConvModel, self).__init__()
        self.emb = nn.Conv2d(3, dims[0], kernel_size=5, padding=2)

        layers = []
        for i in range(len(dims) - 1):
            layers += [
                Block(dims[i], norm_eps=norm_eps) for _ in range(depths[i])
            ]
            layers += [
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=4, stride=2, padding=1)
            ]

        layers += [
            Block(dims[-1], norm_eps=norm_eps) for _ in range(depths[-1])
        ]
        self.layers = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        self.winner_head = nn.Sequential(
            nn.LayerNorm(dims[-1], eps=norm_eps),
            nn.GELU(),
            nn.Linear(dims[-1], 3)
        )

        self.action_head = nn.Sequential(
            nn.LayerNorm(dims[-1], eps=norm_eps),
            nn.GELU(),
            nn.Linear(dims[-1], NUM_COLS)
        )

    def forward(self, board: FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
        x = self.emb(board.permute(0, 3, 1, 2))
        x = self.layers(x)

        return self.winner_head(x), self.action_head(x)
