import tempfile
from pathlib import Path
from typing import Dict

import grpc
import numpy as np

from arachne.runtime.rpc.protobuf import (
    onnxruntime_pb2,
    onnxruntime_pb2_grpc,
    stream_data_pb2,
)
from arachne.runtime.rpc.util.nparray import (
    generator_to_np_array,
    nparray_piece_generator,
)

from .client import RuntimeClientBase


class ONNXRuntimeClient(RuntimeClientBase):
    def __init__(self, channel: grpc.Channel, model_path: str, providers=[]):
        super().__init__(channel)
        # 'provider_options' and 'session_options' are not supported
        self.stub = onnxruntime_pb2_grpc.ONNXRuntimeStub(channel)
        upload_response = self.file_stub_mgr.upload(Path(model_path))
        req = onnxruntime_pb2.InitRequest(model_path=upload_response.filepath, providers=providers)
        self.stub.Init(req)

    def run(self):
        req = onnxruntime_pb2.RunRequest()
        self.stub.Run(req)

    def set_input(self, idx: int, value: np.ndarray):
        def request_generator(idx, value):
            yield onnxruntime_pb2.SetInputRequest(index=idx)

            for piece in nparray_piece_generator(value):
                chunk = stream_data_pb2.Chunk(buffer=piece)
                yield onnxruntime_pb2.SetInputRequest(np_arr_chunk=chunk)

        self.stub.SetInput(request_generator(idx, value))

    def get_output(self, index: int) -> np.ndarray:
        req = onnxruntime_pb2.GetOutputRequest(index=index)
        response_generator = self.stub.GetOutput(req)
        byte_extract_func = lambda response: response.np_data
        np_array = generator_to_np_array(response_generator, byte_extract_func)
        assert isinstance(np_array, np.ndarray)
        return np_array

    def benchmark(self, warmup: int = 1, repeat: int = 10, number: int = 1) -> Dict:
        req = onnxruntime_pb2.BenchmarkRequest(warmup=warmup, repeat=repeat, number=number)
        response = self.stub.Benchmark(req)

        return {
            "mean": response.mean_ts,
            "std": response.std_ts,
            "max": response.max_ts,
            "min": response.min_ts,
        }
