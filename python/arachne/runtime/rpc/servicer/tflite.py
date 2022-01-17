import os

import grpc

from arachne.logger import Logger
from arachne.runtime import TFLiteRuntimeModule, init
from arachne.runtime.rpc.protobuf import tfliteruntime_pb2_grpc
from arachne.runtime.rpc.protobuf.msg_response_pb2 import MsgResponse

from .servicer import RuntimeServicerBase, register_runtime_servicer

logger = Logger.logger()


class TfLiteRuntimeServicer(RuntimeServicerBase, tfliteruntime_pb2_grpc.TfLiteRuntimeServicer):
    @staticmethod
    def register_servicer_to_server(server: grpc.Server):
        tfliteruntime_pb2_grpc.add_TfLiteRuntimeServicer_to_server(TfLiteRuntimeServicer(), server)

    @staticmethod
    def get_name():
        return "tflite"

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
        self.module = init(model_file=model_path)
        assert isinstance(self.module, TFLiteRuntimeModule)
        return MsgResponse(msg="Init")


register_runtime_servicer(TfLiteRuntimeServicer)
