from typing import Callable, Union

from tvm._ffi.runtime_ctypes import Device as TVMDevice
from tvm.contrib.graph_executor import GraphModule
from tvm.contrib.tflite_runtime import TFLiteModule
import tvm

from .types import IndexedOrderedDict


class RuntimeModule(object):
    def __init__(self, module: Union[GraphModule, TFLiteModule]):
        assert isinstance(module, GraphModule) or isinstance(
            module, TFLiteModule)
        self.module = module

    def set_inputs(self, inputs: IndexedOrderedDict, tvmdev: TVMDevice):
        if isinstance(self.module, TFLiteModule):
            for i, value in enumerate(inputs.values()):
                tvm_array = tvm.nd.array(value, tvmdev)
                self.module.set_input(i, tvm_array)
        else:
            self.module.set_input(**inputs)

    def run(self):
        if isinstance(self.module, TFLiteModule):
            self.module.invoke()
        else:
            self.module.run()

    def benchmark(self, tvmdev: TVMDevice, repeat: int) -> Callable:
        run_name = "invoke" if isinstance(self.module, TFLiteModule) else "run"
        timer = self.module.module.time_evaluator(
            run_name, tvmdev, 1, repeat=repeat)
        return timer

    def get_outputs(
        self,
        output_info: IndexedOrderedDict
    ) -> IndexedOrderedDict:
        outputs = IndexedOrderedDict()
        for i, name in enumerate(output_info.keys()):
            outputs[name] = self.module.get_output(i).asnumpy()

        return outputs
