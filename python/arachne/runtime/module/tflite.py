from typing import Dict, List, Union

import numpy as np
import tvm
from tvm.contrib import tflite_runtime
from tvm.contrib.tflite_runtime import TFLiteModule

from arachne.pipeline.package.tflite import TFLitePackage
from arachne.runtime.session import create_tvmdev
from arachne.types import IndexedOrderedDict

from .module import RuntimeModule


class TFLiteRuntimeModule(RuntimeModule):
    def __init__(self, package: TFLitePackage, session: tvm.rpc.RPCSession, profile: bool):
        assert isinstance(package, TFLitePackage)
        tvmdev = create_tvmdev("cpu", session)
        runtime_target = "edge_tpu" if package.for_edgetpu else "cpu"
        with open(package.dir / package.model_file, "rb") as model_fin:
            module = tflite_runtime.create(model_fin.read(), tvmdev, runtime_target)

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
