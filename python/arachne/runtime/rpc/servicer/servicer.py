from abc import abstractmethod
from typing import List, Optional, Type

import grpc

from arachne.registry import Registry
from arachne.runtime.module import RuntimeModule
from arachne.runtime.rpc.protobuf import runtime_message_pb2, runtime_message_pb2_grpc
from arachne.runtime.rpc.protobuf.msg_response_pb2 import MsgResponse
from arachne.runtime.rpc.util.nparray import (
    generator_to_np_array,
    nparray_piece_generator,
)


class RuntimeServicerBase(runtime_message_pb2_grpc.RuntimeServicer):
    @staticmethod
    @abstractmethod
    def register_servicer_to_server(server: grpc.Server):
        pass

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass

    def __init__(self):
        self.module: RuntimeModule

    def SetInput(self, request_iterator, context):
        assert self.module
        index = next(request_iterator).index
        if index is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("index should not be None")
            return MsgResponse()

        byte_extract_func = lambda request: request.np_arr_chunk.buffer
        np_arr = generator_to_np_array(request_iterator, byte_extract_func)
        self.module.set_input(index, np_arr)
        return MsgResponse(msg="SetInput")

    def Run(self, request, context):
        assert self.module
        self.module.run()
        return MsgResponse(msg="Run")

    def Benchmark(self, request, context):
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
        assert self.module
        index = request.index
        np_array = self.module.get_output(index)
        for piece in nparray_piece_generator(np_array):
            yield runtime_message_pb2.GetOutputResponse(np_data=piece)


RuntimeServicerRegistry = Registry[str, Type[RuntimeServicerBase]]()


def get_runtime_servicer(key: str) -> Optional[Type[RuntimeServicerBase]]:
    return RuntimeServicerRegistry.get(key)


def register_runtime_servicer(servicer: Type[RuntimeServicerBase]):
    RuntimeServicerRegistry.register(servicer.get_name(), servicer)


def runtime_servicer_list() -> List[str]:
    return RuntimeServicerRegistry.list()
