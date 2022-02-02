from pathlib import Path

import grpc

from arachne.runtime.rpc.protobuf import onnxruntime_pb2, onnxruntime_pb2_grpc

from .client import RuntimeClientBase


class ONNXRuntimeClient(RuntimeClientBase):
    def __init__(self, channel: grpc.Channel, model_path: str, providers=[]):
        """RuntimeClient for onnx

        Args:
            channel (grpc.Channel): channel to connect server
            model_path (str): path to :code:`.onnx` model file
            providers (list, optional): :code:`providers` to set onnxruntime.InferenceSession. Defaults to [].
        """
        stub = onnxruntime_pb2_grpc.ONNXRuntimeStub(channel)
        super().__init__(channel, stub)
        # 'provider_options' and 'session_options' are not supported
        upload_response = self.file_stub_mgr.upload(Path(model_path))
        req = onnxruntime_pb2.ONNXInitRequest(
            model_path=upload_response.filepath, providers=providers
        )
        self.stub.Init(req)
