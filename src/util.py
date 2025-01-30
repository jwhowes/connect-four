import calendar
import os
import time
from dataclasses import dataclass
from typing import Self

import yaml


@dataclass
class Config:
    @classmethod
    def from_yaml(cls, yaml_path: str) -> Self:
        if not os.path.exists(yaml_path):
            return cls()

        with open(yaml_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        if config is None:
            return cls()

        return cls(**{
            k: float(config.get(k, v.default)) if isinstance(v.default, float) else config.get(k, v.default)
            for k, v in cls.__dataclass_fields__.items()
        })


def timestamp() -> str:
    return calendar.timegm(time.gmtime())
