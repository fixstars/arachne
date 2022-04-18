import json
from pathlib import Path

import grpc

from arachne.runtime.rpc.protobuf import runtime_message_pb2, runtime_pb2_grpc

from .client import RuntimeClientBase


class TfliteRuntimeClient(RuntimeClientBase):
    def __init__(self, channel: grpc.Channel, model_path: str, num_threads=1):
        """RuntimeClient for TfLite

        Args:
            channel (grpc.Channel): channel to connect server
            model_path (str): path to :code:`.tflite` model file
            num_threads (int, optional): :code:`num_threads` to set tfliteInterpreter. Defaults to 1.
        """
        stub = runtime_pb2_grpc.RuntimeStub(channel)
        super().__init__(channel, stub)
        response = self.file_stub_mgr.upload(Path(model_path))
        args_json = json.dumps({"model_path": response.filepath, "num_threads": num_threads})
        req = runtime_message_pb2.InitRequest(args_json=args_json)
        self.stub.Init(req)
