from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
import tvm
from tvm._ffi.runtime_ctypes import Device as TVMDevice
from tvm.contrib.graph_executor import GraphModule
from tvm.contrib.tflite_runtime import TFLiteModule
from tvm.runtime.vm import VirtualMachine

from arachne.pipeline.package.package import Package
from arachne.pipeline.package.tflite import TFLitePackage
from arachne.pipeline.package.tvm import TVMPackage
from arachne.types import IndexedOrderedDict


class RuntimeModule(metaclass=ABCMeta):
    """
    An abstract class for arachne runtime modules.
    Each subclass for this class represents a corresponding tvm runtime.
    """

    module: Union[GraphModule, TFLiteModule, VirtualMachine]
    tvmdev: TVMDevice
    package: Package

    @abstractmethod
    def __init__(
        self,
        module: Union[GraphModule, TFLiteModule, VirtualMachine],
        tvmdev: TVMDevice,
        package: Package,
    ):
        pass

    @abstractmethod
    def set_inputs(self, inputs: Union[IndexedOrderedDict, List]):
        """Set inputs to the module

        Parameters
        ----------
        inputs : dict of str to NDArray
        """
        pass

    @abstractmethod
    def run(self):
        """Run forward execution of the graph"""
        pass

    @abstractmethod
    def benchmark(self, repeat: int) -> Dict:
        """Benchmarking the module with dummy inputs

        Parameters
        ----------
        repeats : the number of times to repeat the measurement

        Returns
        -------
        Dict : The information of benchmark results
        """
        pass

    @abstractmethod
    def get_outputs(self, output_info: IndexedOrderedDict) -> IndexedOrderedDict:
        """Get outputs

        Parameters
        ----------
        output_info : a dict containing tensor names to be get

        Returns
        -------
        IndexedOrderdDict : a dict of output tensors
        """
        pass

    def get_input_details(self) -> IndexedOrderedDict:
        """Get the input tesonr information

        Returns
        -------
        IndexedOrderdDict : a dict of tensor name (str) to TensorInfo
        """
        return self.package.input_info.copy()

    def get_output_details(self) -> IndexedOrderedDict:
        """Get the output tesonr information

        Returns
        -------
        IndexedOrderdDict : a dict of tensor name (str) to TensorInfo
        """
        return self.package.output_info.copy()


class TVMRuntimeModule(RuntimeModule):
    """
    A wrapper class for tvm.contrib.graph_executor.GraphModule
    """

    def __init__(self, module: GraphModule, tvmdev: TVMDevice, package: TVMPackage):
        assert isinstance(module, GraphModule)
        self.module: GraphModule = module
        self.tvmdev = tvmdev
        self.package = package

    def set_inputs(self, inputs: Union[IndexedOrderedDict, List]):
        if isinstance(inputs, IndexedOrderedDict):
            self.module.set_input(**inputs)
        elif isinstance(inputs, list):
            for i, input in enumerate(inputs):
                self.module.set_input(key=i, value=input)
        else:
            raise RuntimeError("unreachable")

    def run(self):
        self.module.run()

    def benchmark(self, repeat: int) -> Dict:
        input_tensors = [
            np.random.uniform(-1, 1, size=ispec.shape).astype(ispec.dtype)
            for ispec in self.package.input_info.values()
        ]

        for i, tensor in enumerate(input_tensors):
            self.set_input(i, tensor)

        timer = self.module.module.time_evaluator("run", self.tvmdev, 1, repeat=repeat)

        self.run()

        prof_result = timer()
        times = prof_result.results

        mean_ts = np.mean(times) * 1000
        std_ts = np.std(times) * 1000
        max_ts = np.max(times) * 1000
        min_ts = np.min(times) * 1000

        return {"mean": mean_ts, "std": std_ts, "max": max_ts, "min": min_ts}

    def get_outputs(self, output_info: IndexedOrderedDict) -> IndexedOrderedDict:
        outputs: IndexedOrderedDict = IndexedOrderedDict()
        for i, name in enumerate(output_info.keys()):
            outputs[name] = self.module.get_output(i).asnumpy()

        return outputs

    def set_input(self, key=None, value=None, **params):
        """A wrapper for GraphModule.set_input()

        Parameters
        ----------
        key : int or str
           The input idx/name

        value : the input value.

        params: additional arguments
        """
        self.module.set_input(key, value, **params)

    def get_input(self, idx: int, out=None):
        """A wrapper for GraphModule.get_input()

        Parameters
        ----------
        idx : int
           The input idx

        out : NDArray or None
            The output array container

        Returns
        -------
        NDArray : the input tensor value
        """
        return self.module.get_input(idx, out)

    def get_num_inputs(self):
        """A wrapper for GraphModule.get_num_inputs()"""
        return self.module.get_num_inputs()

    def get_output(self, idx: int, out=None):
        """A wrapper for GraphModule.get_output()

        Parameters
        ----------
        index : int
            The output index

        out : NDArray
            The output array container

        Returns
        -------
        NDArray : the output tensor value
        """
        return self.module.get_output(idx, out)

    def get_num_outputs(self):
        """A wrapper for GraphModule.get_num_outputs()"""
        return self.module.get_num_outputs()


class TFLiteRuntimeModule(RuntimeModule):
    def __init__(self, module: TFLiteModule, tvmdev: TVMDevice, package: TFLitePackage):
        assert isinstance(module, TFLiteModule)
        self.module: TFLiteModule = module
        self.tvmdev = tvmdev
        self.package = package

    def set_inputs(self, inputs: Union[IndexedOrderedDict, List]):
        if isinstance(inputs, IndexedOrderedDict):
            for i, k in enumerate(self.package.input_info.keys()):
                if k in inputs:
                    tvm_array = tvm.nd.array(inputs[k], self.tvmdev)
                    self.module.set_input(i, tvm_array)
        elif isinstance(inputs, list):
            assert len(inputs) == len(self.package.input_info.keys())
            for i, input in enumerate(inputs):
                tvm_array = tvm.nd.array(input, self.tvmdev)
                self.module.set_input(i, tvm_array)
        else:
            raise RuntimeError("unreachable")

    def run(self):
        self.module.invoke()

    def benchmark(self, repeat: int) -> Dict:
        input_tensors = [
            np.random.uniform(-1, 1, size=ispec.shape).astype(ispec.dtype)
            for ispec in self.package.input_info.values()
        ]

        for i, tensor in enumerate(input_tensors):
            self.set_input(i, tensor)

        timer = self.module.module.time_evaluator("invoke", self.tvmdev, 1, repeat=repeat)

        self.run()

        prof_result = timer()
        times = prof_result.results

        mean_ts = np.mean(times) * 1000
        std_ts = np.std(times) * 1000
        max_ts = np.max(times) * 1000
        min_ts = np.min(times) * 1000

        return {"mean": mean_ts, "std": std_ts, "max": max_ts, "min": min_ts}

    def get_outputs(self, output_info: IndexedOrderedDict) -> IndexedOrderedDict:
        outputs: IndexedOrderedDict = IndexedOrderedDict()
        for i, name in enumerate(output_info.keys()):
            outputs[name] = self.module.get_output(i).asnumpy()

        return outputs

    def set_input(self, idx: int, value):
        """A wrapper for TFLiteModule.set_input()"""
        tvm_array = tvm.nd.array(value, self.tvmdev)
        self.module.set_input(idx, tvm_array)

    def get_output(self, idx: int):
        """A wrapper for TFLiteModule.get_output()"""
        return self.module.get_output(idx)

    def set_num_threads(self, num_threads):
        """A wrapper for TFLiteModule.set_num_threads()"""
        self.module.set_num_threads(num_threads)


class TVMVMRuntimeModule(RuntimeModule):
    def __init__(self, module: VirtualMachine, tvmdev: TVMDevice, package: TVMPackage):
        assert isinstance(module, VirtualMachine)
        self.module: VirtualMachine = module
        self.tvmdev = tvmdev
        self.package = package

    def set_inputs(self, inputs: Union[IndexedOrderedDict, List]):
        if isinstance(inputs, IndexedOrderedDict):
            assert len(self.package.input_info.keys()) == len(inputs.keys())
            tvm_input_dict = {}
            for k, v in inputs.items():
                tvm_input_dict[k] = tvm.nd.array(v, self.tvmdev)
            self.module.set_input("main", **tvm_input_dict)
        elif isinstance(inputs, list):
            assert len(inputs) == len(self.package.input_info.keys())
            tvm_inputs = []
            for input in inputs:
                tvm_inputs.append(tvm.nd.array(input, self.tvmdev))
            self.module.set_input("main", *tvm_inputs)
        else:
            raise RuntimeError("unreachable")

    def run(self):
        self.module.invoke_stateful("main")

    def benchmark(self, repeat: int) -> Dict:
        input_tensors = [
            np.random.uniform(-1, 1, size=ispec.shape).astype(ispec.dtype)
            for ispec in self.package.input_info.values()
        ]

        self.set_inputs(input_tensors)

        timer = self.module.module.time_evaluator("invoke_stateful", self.tvmdev, 1, repeat=repeat)
        self.run()

        prof_result = timer("main")
        times = prof_result.results

        # because first execution is slow, calculate average time from the second execution
        first_ts = times[0] * 1000
        mean_ts = np.mean(times[1:]) * 1000
        std_ts = np.std(times[1:]) * 1000
        max_ts = np.max(times[1:]) * 1000
        min_ts = np.min(times[1:]) * 1000

        return {
            "first": first_ts,
            "mean_rest": mean_ts,
            "std_rest": std_ts,
            "max_rest": max_ts,
            "min_rest": min_ts,
        }

    def get_outputs(self, output_info: IndexedOrderedDict) -> IndexedOrderedDict:
        outputs: IndexedOrderedDict = IndexedOrderedDict()
        module_outputs = self.__vmobj_to_list(self.module.get_outputs())
        for i, name in enumerate(self.package.output_info.keys()):
            if name in output_info.keys():
                outputs[name] = module_outputs[i][0]

        return outputs

    def __vmobj_to_list(self, output):
        if isinstance(output, tvm.nd.NDArray):
            return [output.asnumpy()]
        elif isinstance(output, tvm.runtime.container.ADT) or isinstance(output, list):
            return [self.__vmobj_to_list(f) for f in output]
        else:
            raise RuntimeError("Unknown object type: %s" % type(output))
