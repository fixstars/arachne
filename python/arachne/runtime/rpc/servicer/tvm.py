import os

import grpc

from arachne.runtime import init
from arachne.runtime.module.tvm import TVMRuntimeModule
from arachne.runtime.rpc.logger import Logger
from arachne.runtime.rpc.protobuf import tvmruntime_pb2_grpc
from arachne.runtime.rpc.protobuf.msg_response_pb2 import MsgResponse
from arachne.runtime.rpc.utils.nparray import generator_to_np_array

from .factory import RuntimeServicerBaseFactory
from .servicer import RuntimeServicerBase

logger = Logger.logger()


@RuntimeServicerBaseFactory.register("tvm")
class TVMRuntimeServicer(RuntimeServicerBase, tvmruntime_pb2_grpc.TVMRuntimeServicer):
    """Servicer for TVMRuntime"""

    @staticmethod
    def register_servicer_to_server(server: grpc.Server):
        tvmruntime_pb2_grpc.add_TVMRuntimeServicer_to_server(TVMRuntimeServicer(), server)

    def __init__(self):
        pass

    def Init(self, request, context):
        """Initialize TVMRuntimeModule

        Args:
            request : | TfLiteInitRequest
                      | :code:`request.package_path` (str): path to model tar archive on the server side
            context :
        Returns:
            MsgResponse
        """
        package_path = request.package_path
        if package_path is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("model_path should not be None")
            return MsgResponse()
        elif not os.path.exists(package_path):
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details(f"model_path {package_path} does not exist")
            return MsgResponse()

        logger.info("loading " + package_path)
        self.module = init(runtime="tvm", package_tar=package_path)
        assert isinstance(self.module, TVMRuntimeModule)
        return MsgResponse(msg="Init")

    def SetInput(self, request_iterator, context):
        """Set input parameter to runtime module.

        Args:
            request_iterator : | iterator of SetInputRequest
                               | :code:`request_iterator.index.index_i` (int): layer index to set data
                               | :code:`request_iterator.index.index_s` (str): layer name to set data
                               | :code:`request_iterator.np_arr_chunk.buffer` (bytes): byte chunk data of np.ndarray
            context :

        Returns:
            MsgResponse
        """
        assert self.module
        index = next(request_iterator).index
        # select index from 'oneof' structure
        index = index.index_i if index.index_i is not None else index.index_s
        if index is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("index should not be None")
            return MsgResponse()

        def byte_extract_func(request):
            return request.np_arr_chunk.buffer

        np_arr = generator_to_np_array(request_iterator, byte_extract_func)
        self.module.set_input(index, np_arr)
        return MsgResponse(msg="SetInput")
