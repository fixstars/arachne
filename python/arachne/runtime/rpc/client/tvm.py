from pathlib import Path
from typing import Union

import grpc
import numpy as np

from arachne.runtime.rpc.protobuf import (
    stream_data_pb2,
    tvmruntime_pb2,
    tvmruntime_pb2_grpc,
)
from arachne.runtime.rpc.util.nparray import nparray_piece_generator

from .client import RuntimeClientBase


class TVMRuntimeClient(RuntimeClientBase):
    def __init__(
        self,
        channel: grpc.Channel,
        package_path: str,
        **kwargs,
    ):
        stub = tvmruntime_pb2_grpc.TVMRuntimeStub(channel)
        super().__init__(channel, stub)
        upload_response = self.file_stub_mgr.upload(Path(package_path))
        req = tvmruntime_pb2.TVMInitRequest(package_path=upload_response.filepath)
        self.stub.Init(req)

    def set_input(self, idx: Union[int, str], value: np.ndarray):
        def request_generator(idx, value):
            if isinstance(idx, int):
                idx = tvmruntime_pb2.Index(index_i=idx)
            elif isinstance(idx, str):
                idx = tvmruntime_pb2.Index(index_s=idx)
            yield tvmruntime_pb2.TVMSetInputRequest(index=idx)

            for piece in nparray_piece_generator(value):
                chunk = stream_data_pb2.Chunk(buffer=piece)
                yield tvmruntime_pb2.TVMSetInputRequest(np_arr_chunk=chunk)

        self.stub.SetInput(request_generator(idx, value))
