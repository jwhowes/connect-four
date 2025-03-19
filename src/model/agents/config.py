from typing import Literal, Tuple, Dict, Type
from dataclasses import dataclass

from ...config import Config

from .model import Agent, ViTAgent, ConvAgent


class ModelConfig(Config):
    def __init__(self, arch: str, **kwargs):
        if arch not in agents_lookup:
            raise NotImplementedError(f"Architecture {arch} not found.")

        self.__class__ = agents_lookup[arch]
        agents_lookup[arch].__init__(self, **kwargs)

    def build(self) -> Agent:
        ...


@dataclass
class ViTConfig(ModelConfig):
    arch: Literal["vit"] = "vit"
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    patch_size: int = 2
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0

    def build(self) -> ViTAgent:
        return ViTAgent(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            patch_size=self.patch_size,
            attn_dropout=self.attn_dropout,
            ffn_dropout=self.ffn_dropout
        )


@dataclass
class ConvConfig(ModelConfig):
    arch: Literal["conv"] = "conv"
    dims: Tuple[int, ...] = (96, 192)
    depths: Tuple[int, ...] = (3, 9)
    dropout: float = 0.0

    def build(self) -> ConvAgent:
        return ConvAgent(
            dims=self.dims,
            depths=self.depths,
            dropout=self.dropout
        )


agents_lookup: Dict[str, Type[ModelConfig]] = {
    "vit": ViTConfig,
    "conv": ConvConfig
}
