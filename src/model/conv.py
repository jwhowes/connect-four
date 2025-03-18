from typing import Optional

from torch import nn, Tensor

from .util import RMSNorm, SwiGLU


class ConvBlock(nn.Module):
    def __init__(
            self, d_model: int, kernel_size: int = 7, dropout: float = 0.0,
            d_hidden: Optional[int] = None, norm_eps: float = 1e-6
    ):
        super(ConvBlock, self).__init__()
        self.dwconv = nn.Sequential(
            RMSNorm(d_model, eps=norm_eps, conv=True),
            nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size // 2), groups=d_model)
        )

        self.ffn = nn.Sequential(
            RMSNorm(d_model, eps=norm_eps, conv=True),
            SwiGLU(d_model, d_hidden, conv=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dwconv(x)

        return x + self.ffn(x)
