import os

from arachne.runtime.rpc.protobuf import tfliteruntime_pb2, tfliteruntime_pb2_grpc
from arachne.runtime.rpc.protobuf.msg_response_pb2 import MsgResponse
from arachne.runtime.rpc.util.nparray import (
    generator_to_np_array,
    nparray_piece_generator,
)


class TfLiteRuntimeServicer(tfliteruntime_pb2_grpc.TfliteRuntimeServerServicer):
    def __init__(self):
        self.interpreter = None

    def Init(self, request, context):

        import tensorflow as tf

        model_path = request.model_path
        print("loading ", model_path)
        self.interpreter = tf.lite.Interpreter(model_path, num_threads=request.num_threads)
        self.interpreter.allocate_tensors()
        return MsgResponse(error=False, message="OK")

    def SetInput(self, request_iterator, context):
        assert self.interpreter
        input_details = self.interpreter.get_input_details()
        index = None
        np_arr = None
        byte_extract_func = lambda request: request.np_arr_chunk.buffer
        index = next(request_iterator).index
        assert index is not None
        np_arr = generator_to_np_array(request_iterator, byte_extract_func)

        input_index = input_details[index]["index"]
        self.interpreter.set_tensor(input_index, np_arr)
        return MsgResponse(error=False, message="OK")

    def Invoke(self, request, context):
        assert self.interpreter
        self.interpreter.invoke()
        return MsgResponse(error=False, message="OK")

    def GetOutput(self, request, context):
        index = request.index
        assert self.interpreter
        output_details = self.interpreter.get_output_details()
        np_array = self.interpreter.get_tensor(output_details[index]["index"])
        for piece in nparray_piece_generator(np_array):
            yield tfliteruntime_pb2.GetOutputResponse(np_data=piece)
