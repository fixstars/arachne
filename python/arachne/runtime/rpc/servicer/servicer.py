import json
import os
import tarfile
import tempfile

import grpc

import arachne.runtime
from arachne.runtime.module import RuntimeModuleBase
from arachne.runtime.rpc.logger import Logger
from arachne.runtime.rpc.protobuf import runtime_message_pb2, runtime_pb2_grpc
from arachne.runtime.rpc.protobuf.msg_response_pb2 import MsgResponse
from arachne.runtime.rpc.utils.nparray import (
    generator_to_np_array,
    nparray_piece_generator,
)

logger = Logger.logger()


class RuntimeServicer(runtime_pb2_grpc.RuntimeServicer):
    """runtime servicer"""

    def __init__(self):
        self.module: RuntimeModuleBase  #: runtime module for inference

    def Init(self, request, context):
        """initialize the runtime module."""
        runtime = request.runtime
        args = json.loads(request.args_json)

        path = None
        if "package_tar" in args:
            path = args["package_tar"]
        elif "model_file" in args:
            path = args["model_file"]
        elif "model_dir" in args:
            path = args["model_dir"]
            _, tmpdir = tempfile.mkdtemp()
            with tarfile.open(path, "r") as f:
                f.extractall(tmpdir)
            args["model_dir"] = tempfile

        if path is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                "At leaset one of package_tar or model_file or model_dir must not be None"
            )
            return MsgResponse()
        elif not os.path.exists(path):
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details(f"path {path} does not exist")
            return MsgResponse()
        logger.info("loading " + path)

        self.module = arachne.runtime.init(runtime=runtime, **args)
        return MsgResponse(msg=f"Init {runtime} runtime")

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
        # select index from 'oneof' structure
        index = index.index_i if index.index_i is not None else index.index_s
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
