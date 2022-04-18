import json
from pathlib import Path
from typing import Union

import grpc
import numpy as np

from arachne.runtime.rpc.protobuf import (
    runtime_message_pb2,
    runtime_pb2_grpc,
    stream_data_pb2,
)
from arachne.runtime.rpc.utils.nparray import nparray_piece_generator

from .client import RuntimeClientBase


class TVMRuntimeClient(RuntimeClientBase):
    def __init__(
        self,
        channel: grpc.Channel,
        package_path: str,
        **kwargs,
    ):
        """RuntimeClient for tvm

        Args:
            channel (grpc.Channel): channel to connect server
            package_path (str): path to :code:`.tar` package file
        """
        stub = runtime_pb2_grpc.RuntimeStub(channel)
        super().__init__(channel, stub)
        upload_response = self.file_stub_mgr.upload(Path(package_path))
        package_path = upload_response.filepath
        args_json = json.dumps({"package_path": package_path})
        req = runtime_message_pb2.InitRequest(args_json=args_json)
        self.stub.Init(req)

    def set_input(self, idx: Union[int, str], value: np.ndarray):
        """Request to set input parameter.

        Args:
            idx (Union[int, str]): layer index or layer name to set data
            value (np.ndarray): input data
        """

        def request_generator(idx, value):
            if isinstance(idx, int):
                idx = runtime_message_pb2.Index(index_i=idx)
            elif isinstance(idx, str):
                idx = runtime_message_pb2.Index(index_s=idx)
            yield runtime_message_pb2.SetInputRequest(index=idx)

            for piece in nparray_piece_generator(value):
                chunk = stream_data_pb2.Chunk(buffer=piece)
                yield runtime_message_pb2.SetInputRequest(np_arr_chunk=chunk)

        self.stub.SetInput(request_generator(idx, value))
