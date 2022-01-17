import os

import grpc

from arachne.logger import Logger
from arachne.runtime import ONNXRuntimeModule, init
from arachne.runtime.rpc.protobuf import onnxruntime_pb2_grpc
from arachne.runtime.rpc.protobuf.msg_response_pb2 import MsgResponse

from .servicer import RuntimeServicerBase, register_runtime_servicer

logger = Logger.logger()


class ONNXRuntimeServicer(RuntimeServicerBase, onnxruntime_pb2_grpc.ONNXRuntimeServicer):
    @staticmethod
    def register_servicer_to_server(server: grpc.Server):
        onnxruntime_pb2_grpc.add_ONNXRuntimeServicer_to_server(ONNXRuntimeServicer(), server)

    @staticmethod
    def get_name():
        return "onnx"

    def __init__(self):
        pass

    def Init(self, request, context):
        model_path = request.model_path
        if model_path is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("model_path should not be None")
            return MsgResponse()
        elif not os.path.exists(model_path):
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details(f"model_path {model_path} does not exist")
            return MsgResponse()

        logger.info("loading " + model_path)
        self.module = init(model_file=request.model_path, providers=request.providers)
        assert isinstance(self.module, ONNXRuntimeModule)
        return MsgResponse(msg="Init")


register_runtime_servicer(ONNXRuntimeServicer)
