from math import sqrt
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6, conv: bool = False):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.dim = -1 if not conv else 1

        if not conv:
            self.weight = nn.Parameter(
                torch.empty(d_model).normal_(mean=1.0, std=sqrt(1 / d_model))
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(1, d_model, 1, 1).normal_(mean=1.0, std=sqrt(1 / d_model))
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(self.dim, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: Optional[int] = None, conv: bool = False):
        super(SwiGLU, self).__init__()
        if d_hidden is None:
            d_hidden = 4 * d_model

        if not conv:
            self.gate = nn.Linear(d_model, d_hidden, bias=False)
            self.hidden = nn.Linear(d_model, d_hidden, bias=False)
            self.out = nn.Linear(d_hidden, d_model, bias=False)
        else:
            self.gate = nn.Conv2d(d_model, d_hidden, kernel_size=1, bias=False)
            self.hidden = nn.Conv2d(d_model, d_hidden, kernel_size=1, bias=False)
            self.out = nn.Conv2d(d_hidden, d_model, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(
            F.silu(self.gate(x)) * self.hidden(x)
        )
