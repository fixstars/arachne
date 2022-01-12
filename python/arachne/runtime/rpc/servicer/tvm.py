import grpc

from arachne.logger import Logger
from arachne.runtime import TVMRuntimeModule, init
from arachne.runtime.rpc.protobuf import tvmruntime_pb2, tvmruntime_pb2_grpc
from arachne.runtime.rpc.protobuf.msg_response_pb2 import MsgResponse
from arachne.runtime.rpc.util.nparray import (
    generator_to_np_array,
    nparray_piece_generator,
)

from .servicer import RuntimeServicerBase, register_runtime_servicer

logger = Logger.logger()


class TVMRuntimeServicer(RuntimeServicerBase, tvmruntime_pb2_grpc.TVMRuntimeServerServicer):
    @staticmethod
    def register_servicer_to_server(server: grpc.Server):
        tvmruntime_pb2_grpc.add_TVMRuntimeServerServicer_to_server(TVMRuntimeServicer(), server)

    @staticmethod
    def get_name():
        return "tvm"

    def __init__(self):
        pass

    def Init(self, request, context):
        package_path = request.package_path
        logger.info("loading " + package_path)
        self.module = init(package_path)
        assert isinstance(self.module, TVMRuntimeModule)
        return MsgResponse(msg="OK")

    def SetInput(self, request_iterator, context):
        assert self.module
        index = next(request_iterator).index
        # select index from 'oneof' structure
        index = index.index_i if index.index_i is not None else index.index_s
        assert index is not None, "index should not be None"
        byte_extract_func = lambda request: request.np_arr_chunk.buffer
        np_arr = generator_to_np_array(request_iterator, byte_extract_func)
        self.module.set_input(index, np_arr)
        return MsgResponse(msg="OK")

    def Run(self, request, context):
        assert self.module
        self.module.run()
        return MsgResponse(msg="OK")

    def Benchmark(self, request, context):
        assert self.module
        warmup = request.warmup
        repeat = request.repeat
        number = request.number
        assert isinstance(warmup, int)
        assert isinstance(repeat, int)
        assert isinstance(number, int)
        benchmark_result = self.module.benchmark(warmup=warmup, repeat=repeat, number=number)
        return tvmruntime_pb2.BenchmarkResponse(
            mean_ts=benchmark_result["mean"],
            std_ts=benchmark_result["std"],
            max_ts=benchmark_result["max"],
            min_ts=benchmark_result["min"],
        )

    def GetOutput(self, request, context):
        index = request.index
        assert self.module
        np_array = self.module.get_output(index)
        for piece in nparray_piece_generator(np_array):
            yield tvmruntime_pb2.GetOutputResponse(np_data=piece)


register_runtime_servicer(TVMRuntimeServicer)
