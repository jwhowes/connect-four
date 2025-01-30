from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type, Tuple, Dict

import yaml

from .base import BaseModel
from .conv import ConvModel
from .transformer import Transformer


@dataclass
class BaseModelConfig(ABC):
    @staticmethod
    @abstractmethod
    def model_type() -> Type[BaseModel]:
        ...

    def build_model(self) -> BaseModel:
        return self.model_type()(**self.__dict__)

    @staticmethod
    def from_yaml(yaml_path: str) -> BaseModelConfig:
        with open(yaml_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        assert config is not None and "arch" in config, "Architecture"

        try:
            cls = config_lookup[config["arch"]]
        except KeyError:
            raise NotImplementedError

        return cls(**{
            k: float(config.get(k, v.default)) if isinstance(v.default, float) else config.get(k, v.default)
            for k, v in cls.__dataclass_fields__.items()
        })


@dataclass
class ConvModelConfig(BaseModelConfig):
    dims: Tuple[int, ...] = (64,)
    depths: Tuple[int, ...] = (3,)
    norm_eps: float = 1e-6

    @staticmethod
    def model_type() -> Type[BaseModel]:
        return ConvModel


@dataclass
class TransformerConfig(BaseModelConfig):
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    norm_eps: float = 1e-6

    @staticmethod
    def model_type() -> Type[BaseModel]:
        return Transformer


config_lookup: Dict[str, Type[BaseModelConfig]] = {
    "conv": ConvModelConfig,
    "transformer": TransformerConfig
}
