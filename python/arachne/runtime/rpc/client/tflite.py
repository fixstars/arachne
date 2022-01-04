import numpy as np

from arachne.runtime.rpc.protobuf import (
    stream_data_pb2,
    tfliteruntime_pb2,
    tfliteruntime_pb2_grpc,
)
from arachne.runtime.rpc.util.nparray import (
    generator_to_np_array,
    nparray_piece_generator,
)

from .file import FileClient


class TfliteRuntimeClient():
    def __init__(self, channel):
        self.fileclient = FileClient(channel)
        self.stub = tfliteruntime_pb2_grpc.TfliteRuntimeServerStub(channel)

    def init(self, model_path: str, num_threads=1):
        response = self.fileclient.upload(model_path)
        req = tfliteruntime_pb2.InitRequest(model_path=response.filepath, num_threads=num_threads)
        response = self.stub.Init(req)

    def set_input(self, index: int, np_arr: np.ndarray):
        def request_generator(index, np_arr):
            yield tfliteruntime_pb2.SetInputRequest(index=index)
            for piece in nparray_piece_generator(np_arr):
                chunk = stream_data_pb2.Chunk(buffer=piece)
                yield tfliteruntime_pb2.SetInputRequest(np_arr_chunk=chunk)
        response = self.stub.SetInput(request_generator(index, np_arr))

    def invoke(self):
        # TODO: how to handle timeout?
        req = tfliteruntime_pb2.InvokeRequest()
        response = self.stub.Invoke(req)

    def get_output(self, index: int) -> np.ndarray:
        req = tfliteruntime_pb2.GetOutputRequest(index=index)
        response_generator = self.stub.GetOutput(req)
        print(response_generator)
        byte_extract_func = lambda response: response.np_data
        np_array = generator_to_np_array(response_generator, byte_extract_func)
        assert(isinstance(np_array, np.ndarray))
        return np_array
