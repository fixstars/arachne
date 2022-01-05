import tempfile

import numpy as np
import torch
import torch.onnx
import torchvision
from omegaconf.dictconfig import DictConfig

import arachne.runtime
import arachne.tools.tvm
from arachne.data import Model
from arachne.tools.tvm import TVMConfig
from arachne.utils import get_model_spec, save_model


def test_tvm_runtime():
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
        model, tvmcfg = compile()
        cfg = DictConfig({"tools": {"tvm": tvmcfg}})
        save_model(model, package_path, cfg)
        rtmodule = arachne.runtime.init(package=package_path)
        assert rtmodule
        rtmodule.benchmark()


def test_tflite_runtime():
    import tensorflow as tf

    def save_tflite_model(model_path):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.float32),
            ]
        )
        def add(x, y):
            return x + y

        concrete_func = add.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        tflite_model = converter.convert()
        with open(model_path, "wb") as w:
            w.write(tflite_model)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = tmp_dir + "/model.tflite"
        save_tflite_model(model_path)
        rtmodule = arachne.runtime.init(model=model_path)
        assert rtmodule
        input_data = np.array(1.0, dtype=np.float32)
        rtmodule.set_input(0, input_data)
        rtmodule.set_input(1, input_data)
        rtmodule.run()
        local_output = rtmodule.get_output(0)
        desire_output = np.array(2.0, dtype=np.float32)
        np.testing.assert_equal(local_output, desire_output)
