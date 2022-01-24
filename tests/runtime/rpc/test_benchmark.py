import tempfile

import numpy as np
from tvm.contrib.download import download

import arachne.runtime
import arachne.runtime.rpc
import arachne.tools.tvm
from arachne.runtime.rpc import (
    ONNXRuntimeClient,
    TfliteRuntimeClient,
    TVMRuntimeClient,
    create_server,
)


def test_tvm_runtime_rpc_benchmark(rpc_port=5051):
    with tempfile.TemporaryDirectory() as tmp_dir:
        url = "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/tvm_mobilenet.tar"
        tvm_package_path = tmp_dir + "/tvm_mobilenet.tar"
        download(url, tvm_package_path)

        # rpc run
        server = create_server("tvm", rpc_port)
        server.start()
        try:
            client = arachne.runtime.rpc.init(package_tar=tvm_package_path, rpc_port=rpc_port)
            assert isinstance(client, TVMRuntimeClient)
            client.benchmark()
            client.finalize()
        finally:
            server.stop(0)


def test_tflite_runtime_rpc_benchmark(rpc_port=5051):
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = tmp_dir + "/model.tflite"
        url = "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/mobilenet.tflite"
        download(url, model_path)

        # rpc
        server = create_server("tflite", rpc_port)
        server.start()
        try:
            client = arachne.runtime.rpc.init(model_file=model_path, rpc_port=rpc_port)
            assert isinstance(client, TfliteRuntimeClient)
            client.benchmark()
            client.finalize()
        finally:
            server.stop(0)


def test_onnx_runtime_rpc_benchmark(rpc_port=5051):
    with tempfile.TemporaryDirectory() as tmp_dir:
        url = (
            "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/resnet18.onnx"
        )
        model_path = tmp_dir + "/resnet18.onnx"
        download(url, model_path)
        server = create_server("onnx", rpc_port)
        server.start()
        try:
            ort_opts = {"providers": ["CPUExecutionProvider"]}
            client = arachne.runtime.rpc.init(model_file=model_path, rpc_port=rpc_port, **ort_opts)
            assert isinstance(client, ONNXRuntimeClient)
            client.benchmark()
            client.finalize()
        finally:
            server.stop(0)
