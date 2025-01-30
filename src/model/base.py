from __future__ import annotations

from torch import nn, FloatTensor, Tensor
from abc import ABC, abstractmethod
from typing import Tuple, Self

from ..util import Config


class BaseModel(ABC, nn.Module):
    @classmethod
    def from_config(cls, config: Config) -> Self:
        return cls(
            **config.__dict__
        )

    @abstractmethod
    def forward(self, board: Tensor) -> Tuple[FloatTensor, FloatTensor]:
        ...
