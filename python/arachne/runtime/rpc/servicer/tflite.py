import os

import grpc

from arachne.runtime import init
from arachne.runtime.rpc.logger import Logger
from arachne.runtime.rpc.protobuf import tfliteruntime_pb2_grpc
from arachne.runtime.rpc.protobuf.msg_response_pb2 import MsgResponse

from .factory import RuntimeServicerBaseFactory
from .servicer import RuntimeServicerBase

logger = Logger.logger()


@RuntimeServicerBaseFactory.register("tflite")
class TfLiteRuntimeServicer(RuntimeServicerBase, tfliteruntime_pb2_grpc.TfLiteRuntimeServicer):
    """Servicer for TfLiteRuntime"""

    @staticmethod
    def register_servicer_to_server(server: grpc.Server):
        tfliteruntime_pb2_grpc.add_TfLiteRuntimeServicer_to_server(TfLiteRuntimeServicer(), server)

    def __init__(self):
        pass

    def Init(self, request, context):
        """Initialize TfLiteRuntimeModule

        Args:
            request : | TfLiteInitRequest
                      | :code:`request.model_path` (str): path to the tflite file on the server side
                      | :code:`request.num_threads` (int): num_threads to set tfliteInterpreter. Defaults to 1.
            context :
        Returns:
            MsgResponse
        """
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
        self.module = init(runtime="tflite", model_file=model_path)
        return MsgResponse(msg="Init")
