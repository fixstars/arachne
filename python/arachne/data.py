from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TensorSpec:
    name: str
    shape: List[int]
    dtype: str


@dataclass
class ModelSpec:
    inputs: List[TensorSpec]
    outputs: List[TensorSpec]


@dataclass
class Model:
    path: str
    spec: Optional[ModelSpec] = None
