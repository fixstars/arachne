from dataclasses import dataclass
from typing import Tuple

@dataclass
class InputSpec:
    shape: Tuple[int]
    dtype: str
