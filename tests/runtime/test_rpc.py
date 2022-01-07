import tempfile
from threading import local

import numpy as np
import torch
import torch.onnx
import torchvision
from omegaconf import OmegaConf

import arachne.runtime
import arachne.tools.tvm
from arachne.data import Model
from arachne.runtime.rpc import (
    ONNXRuntimeClient,
    TfliteRuntimeClient,
    TVMRuntimeClient,
    create_channel,
)
from arachne.server import create_server
from arachne.tools.tvm import TVMConfig
from arachne.utils import get_model_spec, save_model


def test_tvm_runtime_rpc(rpc_port=5051):
    def compile():
        cfg = TVMConfig()
        cfg.cpu_target = "x86-64"
        cfg.composite_target = ["cpu"]

        resnet18 = torchvision.models.resnet18(pretrained=True)
        dummy_input = torch.randn(1, 3, 224, 224)
        onnx_model_path = "./resnet18.onnx"
        torch.onnx.export(resnet18, dummy_input, onnx_model_path)
        input = Model("./resnet18.onnx", spec=get_model_spec("resnet18.onnx"))
        model = arachne.tools.tvm.run(input=input, cfg=cfg)
        return model, cfg

    with tempfile.TemporaryDirectory() as tmp_dir:
        package_path = tmp_dir + "/package.tar"
        model, cfg = compile()
        save_model(model, package_path, tvm_cfg=OmegaConf.structured(cfg))

        dummy_input = np.random.rand(1, 3, 224, 224)

        # local run
        rtmodule = arachne.runtime.init(package_tar=package_path)
        assert rtmodule
        rtmodule.set_input(0, dummy_input)
        rtmodule.run()
        local_output = rtmodule.get_output(0)

        # rpc run
        server = create_server(rpc_port)
        server.start()
        try:
            channel = create_channel(port=rpc_port)
            client = TVMRuntimeClient(channel, package_path)
            client.set_input(0, dummy_input)
            client.run()
            rpc_output = client.get_output(0)
        finally:
            server.stop(0)
        # compare
        np.testing.assert_equal(local_output, rpc_output)


def test_tflite_runtime_rpc(rpc_port=5051):
    import tensorflow as tf

    def save_tflite_model(model_path):
        model: tf.keras.Model = tf.keras.applications.mobilenet.MobileNet()
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(model_path, "wb") as w:
            w.write(tflite_model)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = tmp_dir + "/model.tflite"
        save_tflite_model(model_path)
        rtmodule = arachne.runtime.init(model_file=model_path)
        assert rtmodule

        # local
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        rtmodule.set_input(0, dummy_input)
        rtmodule.run()
        local_output = rtmodule.get_output(0)

        # rpc
        server = create_server(rpc_port)
        server.start()
        try:
            channel = create_channel(port=rpc_port)
            client = TfliteRuntimeClient(channel, model_path)
            client.set_input(0, dummy_input)
            client.invoke()
            rpc_output = client.get_output(0)
        finally:
            server.stop(0)

        # compare
        np.testing.assert_allclose(local_output, rpc_output, rtol=1e-5, atol=1e-5)


def test_onnx_runtime_rpc(rpc_port=5051):
    def save_onnx_model(model_path):
        resnet18 = torchvision.models.resnet18(pretrained=True)
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(resnet18, dummy_input, model_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = "./resnet18.onnx"
        save_onnx_model(model_path)

        # dummy_input = np.random.rand(1, 3, 224, 224)
        dummy_input = np.array(np.random.random_sample([1, 3, 224, 224]), dtype=np.float32)  # type: ignore

        # local run
        rtmodule = arachne.runtime.init(model_file=model_path)
        assert rtmodule
        rtmodule.set_input(0, dummy_input)
        rtmodule.run()
        local_output = rtmodule.get_output(0)

        # rpc run
        server = create_server(rpc_port)
        server.start()
        try:
            channel = create_channel(port=rpc_port)
            ort_opts = {"providers": ["CPUExecutionProvider"]}
            client = ONNXRuntimeClient(channel, model_path, **ort_opts)
            client.set_input(0, dummy_input)
            client.run()
            rpc_output = client.get_output(0)
        finally:
            server.stop(0)
        # compare
        np.testing.assert_allclose(local_output, rpc_output, rtol=1e-5, atol=1e-5)
