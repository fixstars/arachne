import os

import grpc

from arachne.logger import Logger
from arachne.runtime import TVMRuntimeModule, init
from arachne.runtime.rpc.protobuf import tvmruntime_pb2_grpc
from arachne.runtime.rpc.protobuf.msg_response_pb2 import MsgResponse
from arachne.runtime.rpc.util.nparray import generator_to_np_array

from .servicer import RuntimeServicerBase, register_runtime_servicer

logger = Logger.logger()


class TVMRuntimeServicer(RuntimeServicerBase, tvmruntime_pb2_grpc.TVMRuntimeServicer):
    @staticmethod
    def register_servicer_to_server(server: grpc.Server):
        tvmruntime_pb2_grpc.add_TVMRuntimeServicer_to_server(TVMRuntimeServicer(), server)

    @staticmethod
    def get_name():
        return "tvm"

    def __init__(self):
        pass

    def Init(self, request, context):
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
        self.module = init(package_path)
        assert isinstance(self.module, TVMRuntimeModule)
        return MsgResponse(msg="Init")

    def SetInput(self, request_iterator, context):
        assert self.module
        index = next(request_iterator).index
        # select index from 'oneof' structure
        index = index.index_i if index.index_i is not None else index.index_s
        if index is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("index should not be None")
            return MsgResponse()

        byte_extract_func = lambda request: request.np_arr_chunk.buffer
        np_arr = generator_to_np_array(request_iterator, byte_extract_func)
        self.module.set_input(index, np_arr)
        return MsgResponse(msg="SetInput")


register_runtime_servicer(TVMRuntimeServicer)
