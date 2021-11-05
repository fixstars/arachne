from typing import Dict, List, Union

import numpy as np
import tvm
import tvm.rpc
from tvm.contrib import onnx_runtime
from tvm.contrib.onnx_runtime import ONNXModule

from arachne.runtime.indexed_ordered_dict import IndexedOrderedDict
from arachne.runtime.package import ONNXPackage
from arachne.runtime.session import create_tvmdev

from ._registry import register_module_class
from .module import RuntimeModule


class ONNXRuntimeModule(RuntimeModule):
    """
    A wrapper class for tvm.contrib.onnx_runtime.ONNXModule
    """

    def __init__(self, package: ONNXPackage, session: tvm.rpc.RPCSession, profile: bool, **kwargs):
        assert isinstance(package, ONNXPackage)

        import onnxruntime as ort

        providers = kwargs.get("providers", ort.get_available_providers())
        if "TensorrtExecutionProvider" in providers or "CUDAExecutionProvider" in providers:
            tvmdev = create_tvmdev("cuda", session)
        else:
            tvmdev = create_tvmdev("cpu", session)

        with open(package.dir / package.model_file, "rb") as model_fin:
            module = onnx_runtime.create(model_fin.read(), tvmdev, ";".join(providers))

        self.module: ONNXModule = module
        self.tvmdev = tvmdev
        self.package = package

    def get_name(self) -> str:
        return "onnx_runtime_module"

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
        """A wrapper for ONNXModule.run()"""
        self.module.run()

    def benchmark(self, repeat: int) -> Dict:
        input_tensors = [
            np.random.uniform(0, 1, size=ispec.shape).astype(ispec.dtype)
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

    def set_input(self, idx: int, value):
        """A wrapper for ONNXModule.set_input()"""
        tvm_array = tvm.nd.array(value, self.tvmdev)
        self.module.set_input(idx, tvm_array)

    def get_output(self, idx: int):
        """A wrapper for ONNXModule.get_output()"""
        return self.module.get_output(idx)


register_module_class(ONNXPackage, ONNXRuntimeModule)
