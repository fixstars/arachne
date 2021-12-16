from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class BaseConfig():
    input: str = MISSING
    input_spec: Optional[str] = None
    output: str = MISSING
