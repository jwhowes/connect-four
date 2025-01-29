from torch import nn, FloatTensor, Tensor
from abc import ABC, abstractmethod
from typing import Tuple


class BaseModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, board: Tensor) -> Tuple[FloatTensor, FloatTensor]:
        ...
