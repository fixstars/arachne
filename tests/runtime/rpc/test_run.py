import os
import tempfile

import numpy as np
from tvm.contrib.download import download

import arachne.runtime
import arachne.runtime.rpc
import arachne.tools.tvm
from arachne.runtime.rpc import create_server


def test_tvm_runtime_rpc(rpc_port=5051):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        url = "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/tvm_mobilenet.tar"
        tvm_package_path = tmp_dir + "/tvm_mobilenet.tar"
        download(url, tvm_package_path)

        dummy_input = np.array(np.random.random_sample([1, 224, 224, 3]), dtype=np.float32)  # type: ignore

        # local run
        rtmodule = arachne.runtime.init(runtime="tvm", package_tar=tvm_package_path)
        assert rtmodule
        rtmodule.set_input(0, dummy_input)
        rtmodule.run()
        local_output = rtmodule.get_output(0)

        # rpc run
        server = create_server(rpc_port)
        server.start()
        try:
            client = arachne.runtime.rpc.init(
                runtime="tvm", package_tar=tvm_package_path, rpc_port=rpc_port
            )
            client.set_input(0, dummy_input)
            client.run()
            rpc_output = client.get_output(0)
            client.finalize()
        finally:
            server.stop(0)
        # compare
        np.testing.assert_equal(local_output, rpc_output)


def test_tvm_runtime_rpc2(rpc_port=5051):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        url = "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/tvm_mobilenet.tar"
        tvm_package_path = tmp_dir + "/tvm_mobilenet.tar"
        download(url, tvm_package_path)

        dummy_input = np.array(np.random.random_sample([1, 224, 224, 3]), dtype=np.float32)  # type: ignore
        import tarfile

        model_file = None
        env_file = None
        with tarfile.open(tvm_package_path, "r:gz") as tar:
            for m in tar.getmembers():
                if m.name == "env.yaml":
                    env_file = m.name
                else:
                    model_file = m.name
            tar.extractall(".")
        assert model_file is not None
        assert env_file is not None
        # local run
        rtmodule = arachne.runtime.init(runtime="tvm", model_file=model_file, env_file=env_file)
        assert rtmodule
        rtmodule.set_input(0, dummy_input)
        rtmodule.run()
        local_output = rtmodule.get_output(0)

        # rpc run
        server = create_server(rpc_port)
        server.start()
        try:
            client = arachne.runtime.rpc.init(
                runtime="tvm", model_file=model_file, env_file=env_file, rpc_port=rpc_port
            )
            client.set_input(0, dummy_input)
            client.run()
            rpc_output = client.get_output(0)
            client.finalize()
        finally:
            server.stop(0)
        # compare
        np.testing.assert_equal(local_output, rpc_output)


def test_tflite_runtime_rpc(rpc_port=5051):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        model_path = tmp_dir + "/model.tflite"
        url = "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/mobilenet.tflite"
        download(url, model_path)

        rtmodule = arachne.runtime.init(runtime="tflite", model_file=model_path)
        assert rtmodule

        # local
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)  # type: ignore
        rtmodule.set_input(0, dummy_input)
        rtmodule.run()
        local_output = rtmodule.get_output(0)

        # rpc
        server = create_server(rpc_port)
        server.start()
        try:
            client = arachne.runtime.rpc.init(
                runtime="tflite", model_file=model_path, rpc_port=rpc_port
            )
            client.set_input(0, dummy_input)
            client.run()
            rpc_output = client.get_output(0)
            client.finalize()
        finally:
            server.stop(0)

        # compare
        np.testing.assert_equal(local_output, rpc_output)


def test_onnx_runtime_rpc(rpc_port=5051):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        url = (
            "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/resnet18.onnx"
        )
        model_path = tmp_dir + "/resnet18.onnx"
        download(url, model_path)

        dummy_input = np.array(np.random.random_sample([1, 3, 224, 224]), dtype=np.float32)  # type: ignore
        ort_opts = {"providers": ["CPUExecutionProvider"]}

        # local run
        rtmodule = arachne.runtime.init(runtime="onnx", model_file=model_path, **ort_opts)
        assert rtmodule
        rtmodule.set_input(0, dummy_input)
        rtmodule.run()
        local_output = rtmodule.get_output(0)
        # rpc run
        server = create_server(rpc_port)
        server.start()
        try:
            client = arachne.runtime.rpc.init(
                runtime="onnx", model_file=model_path, rpc_port=rpc_port, **ort_opts
            )
            client.set_input(0, dummy_input)
            client.run()
            rpc_output = client.get_output(0)
            client.finalize()
        finally:
            server.stop(0)
        # compare
        np.testing.assert_equal(local_output, rpc_output)
