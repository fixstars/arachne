import json
import pathlib
import tarfile
import tempfile
import warnings
from typing import Dict

import grpc
import numpy as np

from arachne.runtime.rpc.protobuf import (
    runtime_message_pb2,
    runtime_pb2_grpc,
    stream_data_pb2,
)
from arachne.runtime.rpc.utils.nparray import (
    generator_to_np_array,
    nparray_piece_generator,
)

from .stubmgr import FileStubManager, ServerStatusStubManager


class RuntimeClient:
    """runtime client.

    Method interface is almost the same as arachne.runtime.module.
    """

    def __init__(self, channel: grpc.Channel, runtime: str, **kwargs):
        """

        Args:
            channel (grpc.Channel): channel to connect server
            runtime (str): runtime name of the server
            stub : stub instance of gRPC generated stub class
        """
        self.finalized = False
        self.stats_stub_mgr = ServerStatusStubManager(channel)
        self.stats_stub_mgr.trylock()
        self.file_stub_mgr = FileStubManager(channel)
        self.stub = runtime_pb2_grpc.RuntimeStub(channel)

        if "package_tar" in kwargs:
            package_tar = kwargs["package_tar"]
            upload_response = self.file_stub_mgr.upload(pathlib.Path(package_tar))
            kwargs["package_tar"] = upload_response.filepath
        if "model_file" in kwargs:
            model_file = kwargs["model_file"]
            upload_response = self.file_stub_mgr.upload(pathlib.Path(model_file))
            kwargs["model_file"] = upload_response.filepath
        if "model_dir" in kwargs:
            model_dir = kwargs["model_dir"]
            with tempfile.NamedTemporaryFile() as f, tarfile.open(fileobj=f, mode="x:gz") as tf:
                tf.add(model_dir)
                upload_response = self.file_stub_mgr.upload(pathlib.Path(f.name))
                kwargs["model_dir"] = upload_response.filepath

        args = json.dumps(kwargs)
        req = runtime_message_pb2.InitRequest(runtime=runtime, args_json=args)
        self.stub.Init(req)

    def finalize(self):
        """Request to unlock server."""
        self.stats_stub_mgr.unlock()
        self.finalized = True

    def __del__(self):
        try:
            if not self.finalized:
                self.finalize()
        except grpc.RpcError:
            # when server is already shutdown, fail to unlock server.
            warnings.warn(UserWarning("Failed to unlock server"))

    def set_input(self, idx: int, np_arr: np.ndarray):
        """Requset to set input parameter.

        Args:
            idx (int): layer index to set data
            np_arr (np.ndarray): input data
        """

        def request_generator(idx, np_arr):
            if isinstance(idx, int):
                idx = runtime_message_pb2.Index(index_i=idx)
            elif isinstance(idx, str):
                idx = runtime_message_pb2.Index(index_s=idx)
            yield runtime_message_pb2.SetInputRequest(index=idx)

            for piece in nparray_piece_generator(np_arr):
                chunk = stream_data_pb2.Chunk(buffer=piece)
                yield runtime_message_pb2.SetInputRequest(np_arr_chunk=chunk)

        self.stub.SetInput(request_generator(idx, np_arr))

    def run(self):
        """Request to invoke inference."""
        req = runtime_message_pb2.RunRequest()
        self.stub.Run(req)

    def get_output(self, index: int) -> np.ndarray:
        """Request to get inference output.

        Args:
            index (int): layer index to get output

        Returns:
            np.ndarray: output data
        """
        req = runtime_message_pb2.GetOutputRequest(index=index)
        response_generator = self.stub.GetOutput(req)

        def byte_extract_func(response):
            return response.np_data

        np_array = generator_to_np_array(response_generator, byte_extract_func)
        assert isinstance(np_array, np.ndarray)
        return np_array

    def benchmark(self, warmup: int = 1, repeat: int = 10, number: int = 1) -> Dict:
        """Request to run benchmark.

        Args:
            warmup (int, optional): [description]. Defaults to 1.
            repeat (int, optional): [description]. Defaults to 10.
            number (int, optional): [description]. Defaults to 1.

        Returns:
            Dict: benchmark result. Result dict has ['mean', 'std', 'max', 'min'] as key. Value is time in milisecond.
        """
        req = runtime_message_pb2.BenchmarkRequest(warmup=warmup, repeat=repeat, number=number)
        response = self.stub.Benchmark(req)

        return {
            "mean": response.mean_ts,
            "std": response.std_ts,
            "max": response.max_ts,
            "min": response.min_ts,
        }
