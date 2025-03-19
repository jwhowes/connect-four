import os
from abc import ABC, abstractmethod
from typing import Self, Dict

import yaml


class Config(ABC):
    @classmethod
    def from_yaml(cls, yaml_path: str) -> Self:
        if not os.path.exists(yaml_path):
            return cls()

        with open(yaml_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        return cls(**{
            k: float(v) if cls.__annotations__.get(k, str) == float else v for k, v in config.items()
        })
