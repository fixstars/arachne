import os
from multiprocessing.spawn import import_main_path

import grpc

from arachne.runtime import init
from arachne.runtime.module.onnx import ONNXRuntimeModule
from arachne.runtime.rpc.logger import Logger
from arachne.runtime.rpc.protobuf import onnxruntime_pb2_grpc
from arachne.runtime.rpc.protobuf.msg_response_pb2 import MsgResponse

from .factory import RuntimeServicerBaseFactory
from .servicer import RuntimeServicerBase

logger = Logger.logger()


@RuntimeServicerBaseFactory.register("onnx")
class ONNXRuntimeServicer(RuntimeServicerBase, onnxruntime_pb2_grpc.ONNXRuntimeServicer):
    """Servicer for ONNXRuntime"""

    @staticmethod
    def register_servicer_to_server(server: grpc.Server):
        onnxruntime_pb2_grpc.add_ONNXRuntimeServicer_to_server(ONNXRuntimeServicer(), server)

    def __init__(self):
        pass

    def Init(self, request, context):
        """Initialize ONNXRuntimeModule

        Args:
            request : | ONNXInitRequest
                      | :code:`request.model_path` (str): path to the onnx file on the server side
                      | :code:`request.providers` (List[str]): providers to set onnxruntime.InferenceSession. Defaults to [].
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
        self.module = init(
            runtime="onnx", model_file=request.model_path, providers=request.providers
        )
        assert isinstance(self.module, ONNXRuntimeModule)
        return MsgResponse(msg="Init")
