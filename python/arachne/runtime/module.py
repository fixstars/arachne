from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Union

import numpy as np
import tvm
from tensorflow.python.keras.backend import dtype
from tvm._ffi.runtime_ctypes import Device as TVMDevice
from tvm.contrib.graph_executor import GraphModule
from tvm.contrib.tflite_runtime import TFLiteModule
from tvm.driver import tvmc
from tvm.runtime.vm import VirtualMachine

from arachne.pipeline.package.package import Package
from arachne.pipeline.package.tflite import TFLitePackage
from arachne.pipeline.package.tvm import TVMPackage
from arachne.types import IndexedOrderedDict


class RuntimeModule(metaclass=ABCMeta):
    module: Union[GraphModule, TFLiteModule, VirtualMachine]
    tvmdev: TVMDevice
    package: Package

    @abstractmethod
    def __init__(
        self, module: Union[GraphModule, TFLiteModule, VirtualMachine], tvmdev: TVMDevice, package: Package
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

class TVMVMRuntimeModule(RuntimeModule):
    def __init__(self, module: VirtualMachine, tvmdev: TVMDevice, package: TVMPackage):
        assert isinstance(module, VirtualMachine)
        self.module = module
        self.tvmdev = tvmdev
        self.package = package
    
    def set_input(self, idx: int, tensor: Any):
        raise NotImplementedError("cannot use index to set input to the VM")

    def set_inputs(self, inputs: IndexedOrderedDict):
        self.module.set_input("main", inputs)

    def _vmobj_to_list(self, output):
        if isinstance(output, tvm.nd.NDArray):
            return [output.asnumpy()]
        elif isinstance(output, tvm.runtime.container.ADT) or isinstance(output, list):
            return [self._vmobj_to_list(f) for f in output]
        else:
            raise RuntimeError("Unknown object type: %s" % type(output))

    def get_outputs(self) -> IndexedOrderedDict:
        outputs: IndexedOrderedDict = IndexedOrderedDict()
        module_outputs = self._vmobj_to_list(self.module.get_outputs())
        for i, name in enumerate(self.package.output_info.keys()):
            outputs[name] = module_outputs[i][0]

        return outputs

    def get_output(self, idx: int):
        raise NotImplementedError("cannot retrive VM output tensor by get_output. use get_outputs.")

    def get_num_outputs(self):
        return self.module.get_num_outputs()

    def run(self):
        self.module.invoke_stateful("main")

    def benchmark(self, repeat: int) -> Dict:
        return self._do_benchmark(repeat)
    
    def _do_benchmark(self, repeat: int) -> Dict:
        import time

        input_tensors = [
            np.random.uniform(-1, 1, size=ispec.shape).astype(ispec.dtype)
            for ispec in self.package.input_info.values()
        ]

        self.set_inputs(input_tensors)
        
        elapsed = []
        for _ in range(repeat):
            t1 = time.perf_counter()
            self.run()
            t2 = time.perf_counter()
            elapsed.append(t2 - t1)
        elapsed = np.array(elapsed)

        first_ts = elapsed[0] * 1000
        # because first execution is slow, calculate average time from the second execution
        mean_ts = np.mean(elapsed[1:]) * 1000
        std_ts = np.std(elapsed[1:]) * 1000
        max_ts = np.max(elapsed[1:]) * 1000
        min_ts = np.min(elapsed[1:]) * 1000

        return {"first": first_ts, "mean_rest": mean_ts, "std_rest": std_ts, "max_rest": max_ts, "min_rest": min_ts}
