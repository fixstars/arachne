from . import onnx, tflite, tvm
from .factory import RuntimeModuleBase, RuntimeModuleFactory, logger

# from abc import ABCMeta, abstractmethod
# from typing import Any, Dict, List

# import numpy as np


# class RuntimeModule(metaclass=ABCMeta):
#     """Base class of runtime module.

#     RuntimeModule wraps the runtime of the model framework.
#     """

#     module: Any

#     @abstractmethod
#     def __init__(self, model: str, **kwargs):
#         pass

#     @abstractmethod
#     def run(self):
#         """run inference"""
#         pass

#     @abstractmethod
#     def set_input(self, idx, value, **kwargs):
#         pass

#     @abstractmethod
#     def get_output(self, idx) -> np.ndarray:
#         pass

#     @abstractmethod
#     def get_input_details(self) -> List[dict]:
#         """Get model input details. Dict format depends on the type of the runtime.

#         Returns:
#             List[dict]: List of the model input info.
#         """
#         pass

#     @abstractmethod
#     def get_output_details(self) -> List[dict]:
#         """Get model output details. Dict format depends on the type of the runtime.

#         Returns:
#             List[dict]: List of the model output info.
#         """
#         pass

#     @abstractmethod
#     def benchmark(self, warmup: int = 1, repeat: int = 10, number: int = 1) -> Dict:
#         """Request to run benchmark.

#         Args:
#             warmup (int, optional): [description]. Defaults to 1.
#             repeat (int, optional): [description]. Defaults to 10.
#             number (int, optional): [description]. Defaults to 1.

#         Returns:
#             Dict: benchmark result. Result dict has ['mean', 'std', 'max', 'min'] as key. Value is time in milisecond.
#         """
#         pass
