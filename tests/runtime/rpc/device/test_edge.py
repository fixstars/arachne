import os
import tempfile

import cv2
import numpy as np
import pytest
from tvm.contrib.download import download

import arachne.runtime
import arachne.runtime.rpc
import arachne.tools.tvm
from arachne.runtime.rpc import ONNXRuntimeClient, TfliteRuntimeClient, TVMRuntimeClient


def get_input_data():
    image_url = "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/data/cat.jpg"
    image_path = "cat.jpg"
    download(image_url, image_path)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img - np.array([123.0, 117.0, 104.0])
    img /= np.array([58.395, 57.12, 57.375])
    input_data = img[np.newaxis, :, :, :].astype(np.float32)  # type: ignore
    return input_data


@pytest.mark.edgetest
def test_tvm_runtime_rpc(pytestconfig):
    rpc_port = pytestconfig.getoption("rpc_port")
    rpc_host = pytestconfig.getoption("rpc_host")
    tvm_target_device = pytestconfig.getoption("tvm_target_device")
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        import tensorflow as tf
        from omegaconf import OmegaConf

        from arachne.tools import ToolFactory
        from arachne.tools.tvm import TVMConfig, get_predefined_config
        from arachne.utils.model_utils import init_from_file, save_model

        def compile_model(cfg, package_path):
            input = init_from_file("tmp.h5")
            input.spec.inputs[0].shape = [1, 224, 224, 3]  # type: ignore
            input.spec.outputs[0].shape = [1, 1000]  # type: ignore

            tool = ToolFactory.get("tvm")
            output = tool.run(input=input, cfg=cfg)
            save_model(output, package_path, tvm_cfg=OmegaConf.structured(cfg))

        model = tf.keras.applications.mobilenet.MobileNet()
        model.save("tmp.h5")

        # host compile and run
        cfg = TVMConfig()
        cfg.cpu_target = "x86-64"
        cfg.composite_target = ["cpu"]
        host_package_path = tmp_dir + "/host_mobilenet.tar"
        compile_model(cfg, host_package_path)
        rtmodule = arachne.runtime.init(package_tar=host_package_path)
        assert rtmodule
        input_data = get_input_data()
        rtmodule.set_input(0, input_data)
        rtmodule.run()
        host_output = rtmodule.get_output(0)
        host_result = np.argmax(host_output)

        # edge compile and run
        cfg = get_predefined_config(tvm_target_device)
        edge_package_path = tmp_dir + "/edge_mobilenet.tar"
        compile_model(cfg, edge_package_path)
        client = arachne.runtime.rpc.init(
            package_tar=edge_package_path,
            rpc_host=rpc_host,
            rpc_port=rpc_port,
        )
        assert isinstance(client, TVMRuntimeClient)
        client.set_input(0, input_data)
        client.run()
        edge_output = client.get_output(0)
        edge_result = np.argmax(edge_output)
        client.finalize()
        # compare
        assert host_result == edge_result


@pytest.mark.edgetest
def test_tflite_runtime_rpc(pytestconfig):
    rpc_port = pytestconfig.getoption("rpc_port")
    rpc_host = pytestconfig.getoption("rpc_host")
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        model_path = tmp_dir + "/model.tflite"
        url = "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/mobilenet.tflite"
        download(url, model_path)

        rtmodule = arachne.runtime.init(model_file=model_path)
        assert rtmodule

        # host
        input_data = get_input_data()
        rtmodule.set_input(0, input_data)
        rtmodule.run()
        host_output = rtmodule.get_output(0)
        host_result = np.argmax(host_output)
        print(host_output.shape, host_result)

        # edge
        client = arachne.runtime.rpc.init(
            model_file=model_path,
            rpc_host=rpc_host,
            rpc_port=rpc_port,
        )
        assert isinstance(client, TfliteRuntimeClient)
        client.set_input(0, input_data)
        client.run()
        rpc_output = client.get_output(0)
        rpc_result = np.argmax(rpc_output)
        client.finalize()

        # compare
        assert host_result == rpc_result


@pytest.mark.edgetest
def test_onnx_runtime_rpc(pytestconfig):
    rpc_port = pytestconfig.getoption("rpc_port")
    rpc_host = pytestconfig.getoption("rpc_host")
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        url = (
            "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/resnet18.onnx"
        )
        model_path = tmp_dir + "/resnet18.onnx"
        download(url, model_path)

        input_data = np.transpose(get_input_data(), (0, 3, 1, 2))
        ort_opts = {"providers": ["CPUExecutionProvider"]}

        # host
        rtmodule = arachne.runtime.init(model_file=model_path, **ort_opts)
        assert rtmodule
        rtmodule.set_input(0, input_data)
        rtmodule.run()
        local_output = rtmodule.get_output(0)
        local_result = np.argmax(local_output)
        # edge
        client = arachne.runtime.rpc.init(
            model_file=model_path,
            rpc_host=rpc_host,
            rpc_port=rpc_port,
            **ort_opts,
        )
        assert isinstance(client, ONNXRuntimeClient)
        client.set_input(0, input_data)
        client.run()
        rpc_output = client.get_output(0)
        rpc_result = np.argmax(rpc_output)
        client.finalize()
        # compare
        print(local_output.shape, rpc_output.shape)
        print(local_result, rpc_result)
        assert local_result == rpc_result
