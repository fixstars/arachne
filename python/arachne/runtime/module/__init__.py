from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import numpy as np


class RuntimeModule(metaclass=ABCMeta):
    module: Any
    env: dict

    @abstractmethod
    def __init__(self, model: str, **kwargs):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def set_input(self, idx, value, **kwargs):
        pass

    @abstractmethod
    def get_output(self, idx) -> np.ndarray:
        pass

    @abstractmethod
    def get_input_details(self):
        pass

    @abstractmethod
    def get_output_details(self):
        pass

    @abstractmethod
    def benchmark(self, warmup: int = 1, repeat: int = 10, number: int = 1) -> Dict:
        pass
