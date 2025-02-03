from abc import ABC, abstractmethod
from typing import Tuple

from torch import nn, FloatTensor, Tensor


class BaseModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, board: Tensor) -> Tuple[FloatTensor, FloatTensor]:
        ...
