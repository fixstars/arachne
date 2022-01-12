import grpc

from arachne.runtime import ONNXRuntimeModule, init
from arachne.runtime.rpc.protobuf import onnxruntime_pb2, onnxruntime_pb2_grpc
from arachne.runtime.rpc.protobuf.msg_response_pb2 import MsgResponse
from arachne.runtime.rpc.util.nparray import (
    generator_to_np_array,
    nparray_piece_generator,
)

from .servicer import RuntimeServicerBase, register_runtime_servicer


class ONNXRuntimeServicer(RuntimeServicerBase, onnxruntime_pb2_grpc.ONNXRuntimeServerServicer):
    @staticmethod
    def register_servicer_to_server(server: grpc.Server):
        onnxruntime_pb2_grpc.add_ONNXRuntimeServerServicer_to_server(ONNXRuntimeServicer(), server)

    @staticmethod
    def get_name():
        return "onnx"

    def __init__(self):
        pass

    def Init(self, request, context):
        self.module = init(model_file=request.model_path, providers=request.providers)
        assert isinstance(self.module, ONNXRuntimeModule)
        return MsgResponse(error=False, message="OK")

    def SetInput(self, request_iterator, context):
        assert self.module
        index = next(request_iterator).index
        assert index is not None
        byte_extract_func = lambda request: request.np_arr_chunk.buffer
        np_arr = generator_to_np_array(request_iterator, byte_extract_func)
        self.module.set_input(index, np_arr)
        return MsgResponse(error=False, message="OK")

    def Run(self, request, context):
        assert self.module
        self.module.run()
        return MsgResponse(error=False, message="OK")

    def Benchmark(self, request, context):
        assert self.module
        warmup = request.warmup
        repeat = request.repeat
        number = request.number
        assert isinstance(warmup, int)
        assert isinstance(repeat, int)
        assert isinstance(number, int)
        benchmark_result = self.module.benchmark(warmup=warmup, repeat=repeat, number=number)
        return onnxruntime_pb2.BenchmarkResponse(
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
            yield onnxruntime_pb2.GetOutputResponse(np_data=piece)


register_runtime_servicer(ONNXRuntimeServicer)
