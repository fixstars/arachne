import warnings
from abc import ABCMeta
from typing import Dict

import grpc
import numpy as np

from arachne.runtime.rpc.protobuf import runtime_message_pb2, stream_data_pb2
from arachne.runtime.rpc.util.nparray import (
    generator_to_np_array,
    nparray_piece_generator,
)

from .stubmgr import FileStubManager, ServerStatusStubManager


class RuntimeClientBase(metaclass=ABCMeta):
    def __init__(self, channel: grpc.Channel, stub):
        self.finalized = False
        self.stats_stub_mgr = ServerStatusStubManager(channel)
        self.stats_stub_mgr.trylock()
        self.file_stub_mgr = FileStubManager(channel)
        self.stub = stub

    def finalize(self):
        self.stats_stub_mgr.unlock()
        self.finalized = True

    def __del__(self):
        try:
            if not self.finalized:
                self.finalize()
        except:
            # when server is already shutdown, fail to unlock server.
            warnings.warn(UserWarning("Failed to unlock server"))

    def set_input(self, index: int, np_arr: np.ndarray):
        def request_generator(index, np_arr):
            yield runtime_message_pb2.SetInputRequest(index=index)
            for piece in nparray_piece_generator(np_arr):
                chunk = stream_data_pb2.Chunk(buffer=piece)
                yield runtime_message_pb2.SetInputRequest(np_arr_chunk=chunk)

        self.stub.SetInput(request_generator(index, np_arr))

    def run(self):
        req = runtime_message_pb2.RunRequest()
        self.stub.Run(req)

    def get_output(self, index: int) -> np.ndarray:
        req = runtime_message_pb2.GetOutputRequest(index=index)
        response_generator = self.stub.GetOutput(req)
        byte_extract_func = lambda response: response.np_data
        np_array = generator_to_np_array(response_generator, byte_extract_func)
        assert isinstance(np_array, np.ndarray)
        return np_array

    def benchmark(self, warmup: int = 1, repeat: int = 10, number: int = 1) -> Dict:
        req = runtime_message_pb2.BenchmarkRequest(warmup=warmup, repeat=repeat, number=number)
        response = self.stub.Benchmark(req)

        return {
            "mean": response.mean_ts,
            "std": response.std_ts,
            "max": response.max_ts,
            "min": response.min_ts,
        }
