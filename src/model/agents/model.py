from typing import Tuple
from abc import ABC

from torch import nn, Tensor

from ..util import RMSNorm
from ..conv import ConvBlock
from ..transformer import ViT


class Agent(ABC, nn.Module):
    pass


class ViTAgent(Agent, ViT):
    def __init__(
            self, d_model: int, n_heads: int, n_layers: int, patch_size: int = 2,
            attn_dropout: float = 0.0, ffn_dropout: float = 0.0
    ):
        ViT.__init__(
            self,
            in_channels=3,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            patch_size=patch_size,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout
        )


class ConvAgent(Agent):
    def __init__(self, dims: Tuple[int, ...] = (96, 192), depths: Tuple[int, ...] = (3, 9), dropout: float = 0.0):
        super(ConvAgent, self).__init__()

        conv_layers = [
            nn.Conv2d(3, dims[0], kernel_size=3, padding=1)
        ]

        for i in range(len(dims) - 1):
            conv_layers += [
                ConvBlock(dims[i], dropout=dropout)
                for _ in range(depths[i])
            ] + [
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            ]

        conv_layers += [
            ConvBlock(dims[-1], dropout=dropout)
            for _ in range(depths[-1])
        ]

        self.conv = nn.Sequential(*conv_layers)

        self.head = nn.Sequential(
            RMSNorm(dims[-1]),
            nn.Linear(dims[-1], 1)
        )

    def forward(self, grid: Tensor) -> Tensor:
        return self.head(
            self.conv(grid).mean((2, 3))
        ).squeeze(-1)
