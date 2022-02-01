from pathlib import Path

import grpc

from arachne.runtime.rpc.protobuf import tfliteruntime_pb2, tfliteruntime_pb2_grpc

from .client import RuntimeClientBase


class TfliteRuntimeClient(RuntimeClientBase):
    def __init__(self, channel: grpc.Channel, model_path: str, num_threads=1):
        """RuntimeClient for TfLite

        Args:
            channel (grpc.Channel): channel to connect server
            model_path (str): path to :code:`.tflite` model file
            num_threads (int, optional): :code:`num_threads` to set tfliteInterpreter. Defaults to 1.
        """
        stub = tfliteruntime_pb2_grpc.TfLiteRuntimeStub(channel)
        super().__init__(channel, stub)
        response = self.file_stub_mgr.upload(Path(model_path))
        req = tfliteruntime_pb2.TfLiteInitRequest(
            model_path=response.filepath, num_threads=num_threads
        )
        self.stub.Init(req)
