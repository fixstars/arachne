from abc import ABCMeta, abstractmethod
from logging import getLogger
from typing import Callable, Dict, List

import numpy as np

logger = getLogger(__name__)


class RuntimeModuleBase(metaclass=ABCMeta):
    """Base class of runtime module.

    RuntimeModule wraps the runtime of the model framework.
    """

    @abstractmethod
    def __init__(self, model: str, **kwargs):
        pass

    @abstractmethod
    def run(self):
        """run inference"""
        pass

    @abstractmethod
    def set_input(self, idx, value, **kwargs):
        pass

    @abstractmethod
    def get_output(self, idx, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def get_input_details(self) -> List[dict]:
        """Get model input details. Dict format depends on the type of the runtime.

        Returns:
            List[dict]: List of the model input info.
        """
        pass

    @abstractmethod
    def get_output_details(self) -> List[dict]:
        """Get model output details. Dict format depends on the type of the runtime.

        Returns:
            List[dict]: List of the model output info.
        """
        pass

    @abstractmethod
    def benchmark(self, warmup: int = 1, repeat: int = 10, number: int = 1) -> Dict:
        """Request to run benchmark.

        Args:
            warmup (int, optional): [description]. Defaults to 1.
            repeat (int, optional): [description]. Defaults to 10.
            number (int, optional): [description]. Defaults to 1.

        Returns:
            Dict: benchmark result. Result dict has ['mean', 'std', 'max', 'min'] as key. Value is time in milisecond.
        """
        pass


class RuntimeModuleFactory:
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: RuntimeModuleBase) -> RuntimeModuleBase:
            if name in cls.registry:
                logger.warning(f"RuntimeModule for {name} already exists. Will replace it")
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(cls, name: str, **kwargs) -> RuntimeModuleBase:
        if name not in cls.registry:
            raise Exception(f"RuntimeModule {name} not exists in the registry")
        runtime_class = cls.registry[name]
        runtime = runtime_class(**kwargs)
        return runtime

    @classmethod
    def list(cls) -> List[str]:
        return list(cls.registry.keys())
