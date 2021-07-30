from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Union

import numpy as np
import tvm
from tensorflow.python.keras.backend import dtype
from tvm._ffi.runtime_ctypes import Device as TVMDevice
from tvm.contrib.graph_executor import GraphModule
from tvm.contrib.tflite_runtime import TFLiteModule
from tvm.driver import tvmc

from arachne.pipeline.package.package import Package
from arachne.pipeline.package.tflite import TFLitePackage
from arachne.pipeline.package.tvm import TVMPackage
from arachne.types import IndexedOrderedDict


class RuntimeModule(metaclass=ABCMeta):
    module: Union[GraphModule, TFLiteModule]
    tvmdev: TVMDevice
    package: Package

    @abstractmethod
    def __init__(
        self, module: Union[GraphModule, TFLiteModule], tvmdev: TVMDevice, package: Package
    ):
        pass

    def set_input(self, idx: int, tensor: Any):
        self.module.set_input(idx, tensor)

    @abstractmethod
    def set_inputs(self, inputs: IndexedOrderedDict):
        pass

    def get_input_details(self) -> IndexedOrderedDict:
        return self.package.input_info.copy()

    @abstractmethod
    def run(self):
        pass

    def _do_benchmark(self, repeat: int, run_method: str) -> Dict:
        input_tensors = [
            np.random.uniform(-1, 1, size=ispec.shape).astype(ispec.dtype)
            for ispec in self.package.input_info.values()
        ]

        for i, tensor in enumerate(input_tensors):
            self.set_input(i, tensor)

        timer = self.module.module.time_evaluator(run_method, self.tvmdev, 1, repeat=repeat)

        self.run()

        prof_result = timer()
        times = prof_result.results

        result = tvmc.TVMCResult(None, times)

        mean_ts = np.mean(result.times) * 1000
        std_ts = np.std(result.times) * 1000
        max_ts = np.max(result.times) * 1000
        min_ts = np.min(result.times) * 1000

        return {"mean": mean_ts, "std": std_ts, "max": max_ts, "min": min_ts}

    @abstractmethod
    def benchmark(self, repeat: int) -> Dict:
        pass

    def get_outputs(self, output_info: IndexedOrderedDict) -> IndexedOrderedDict:
        outputs: IndexedOrderedDict = IndexedOrderedDict()
        for i, name in enumerate(output_info.keys()):
            outputs[name] = self.module.get_output(i).asnumpy()

        return outputs

    def get_output_details(self) -> IndexedOrderedDict:
        return self.package.output_info.copy()


class TVMRuntimeModule(RuntimeModule):
    def __init__(self, module: GraphModule, tvmdev: TVMDevice, package: TVMPackage):
        assert isinstance(module, GraphModule)
        self.module = module
        self.tvmdev = tvmdev
        self.package = package

    def set_inputs(self, inputs: IndexedOrderedDict):
        self.module.set_input(**inputs)

    def get_input(self, idx: int, out=None):
        return self.module.get_input(idx, out)

    def get_num_inputs(self):
        return self.module.get_num_inputs()

    def get_output(self, idx: int, out=None):
        return self.module.get_output(idx, out)

    def get_num_outputs(self):
        return self.module.get_num_outputs()

    def run(self):
        self.module.run()

    def benchmark(self, repeat: int) -> Dict:
        return self._do_benchmark(repeat, "run")


class TFLiteRuntimeModule(RuntimeModule):
    def __init__(self, module: TFLiteModule, tvmdev: TVMDevice, package: TFLitePackage):
        assert isinstance(module, TFLiteModule)
        self.module = module
        self.tvmdev = tvmdev
        self.package = package

    def set_input(self, idx: int, value):
        tvm_array = tvm.nd.array(value, self.tvmdev)
        self.module.set_input(idx, tvm_array)

    def set_inputs(self, inputs: IndexedOrderedDict):
        for i, value in enumerate(inputs.values()):
            tvm_array = tvm.nd.array(value, self.tvmdev)
            self.module.set_input(i, tvm_array)

    def get_output(self, idx: int):
        return self.module.get_output(idx)

    def set_num_threads(self, num_threads):
        self.module.set_num_threads(num_threads)

    def run(self):
        self.module.invoke()

    def benchmark(self, repeat: int) -> Dict:
        return self._do_benchmark(repeat, "invoke")
