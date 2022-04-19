from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Type

import grpc

from arachne.runtime.module import RuntimeModuleBase
from arachne.runtime.rpc.protobuf import runtime_message_pb2, runtime_message_pb2_grpc
from arachne.runtime.rpc.protobuf.msg_response_pb2 import MsgResponse
from arachne.runtime.rpc.utils.nparray import (
    generator_to_np_array,
    nparray_piece_generator,
)


class RuntimeServicerBase(runtime_message_pb2_grpc.RuntimeServicer):
    """Base class of runtime servicer"""

    @staticmethod
    @abstractmethod
    def register_servicer_to_server(server: grpc.Server):
        """Register servicer to server using grpc generated function

        :code:`<runtime name>_pb2_grpc.add_<runtime name>RuntimeServicer_to_server`

        Args:
            server(grpc.Server): server to register servicer
        """
        pass

    def __init__(self):
        self.module: RuntimeModuleBase  #: runtime module for inference

    @abstractmethod
    def Init(self, request, context):
        """abstract method to initialize runtime module."""

    def SetInput(self, request_iterator, context):
        """Set input parameter to runtime module.

        Args:
            request_iterator : | iterator of SetInputRequest
                               | :code:`request_iterator.index` (int): layer index to set data
                               | :code:`request_iterator.np_arr_chunk.buffer` (bytes): byte chunk data of np.ndarray
            context :

        Returns:
            MsgResponse
        """
        assert self.module
        index = next(request_iterator).index
        if index is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("index should not be None")
            return MsgResponse()

        def byte_extract_func(request):
            return request.np_arr_chunk.buffer

        np_arr = generator_to_np_array(request_iterator, byte_extract_func)
        self.module.set_input(index, np_arr)
        return MsgResponse(msg="SetInput")

    def Run(self, request, context):
        """Invoke inference on runtime module.

        Args:
            request : SetInputRequest
            context :
        Returns:
            MsgResponse
        """
        assert self.module
        self.module.run()
        return MsgResponse(msg="Run")

    def Benchmark(self, request, context):
        """Run benchmark on runtime module.

        Args:
            request : | BenchmarkRequest
                      | :code:`request.warmup` (int)
                      | :code:`request.repeat` (int)
                      | :code:`request.number` (int)
            context :
        Returns:
            BenchmarkResponse
        """
        assert self.module
        warmup = request.warmup
        repeat = request.repeat
        number = request.number

        if warmup is None or repeat is None or number is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("warmup, repeat, and number should not be None")
            return MsgResponse()

        benchmark_result = self.module.benchmark(warmup=warmup, repeat=repeat, number=number)
        return runtime_message_pb2.BenchmarkResponse(
            mean_ts=benchmark_result["mean"],
            std_ts=benchmark_result["std"],
            max_ts=benchmark_result["max"],
            min_ts=benchmark_result["min"],
        )

    def GetOutput(self, request, context):
        """Get output from runtime module.

        Args:
            request : | GetOutputRequest
                      | :code:`request.index` (int): layer index to get output
            context :
        Returns:
            GetOutputResponse
        """
        assert self.module
        index = request.index
        np_array = self.module.get_output(index)
        for piece in nparray_piece_generator(np_array):
            yield runtime_message_pb2.GetOutputResponse(np_data=piece)


class RuntimeServicerRegistry:
    def __init__(self):
        self._registries: Dict[str, Type[RuntimeServicerBase]] = OrderedDict()

    def register(self, key: str, value: Type[RuntimeServicerBase], override=False):
        assert override or key not in self._registries.keys()
        self._registries[key] = value
        return value

    def get(self, key: str) -> Optional[Type[RuntimeServicerBase]]:
        return self._registries.get(key)

    def list(self) -> List[str]:
        return list(self._registries.keys())
