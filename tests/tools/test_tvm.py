import os
import tempfile

import pytest
import tensorflow as tf
import torch
import torch.jit
import torch.onnx
import torchvision

from arachne.data import Model, ModelSpec, TensorSpec
from arachne.tools.tvm import TVMConfig, run
from arachne.utils import get_model_spec

params = {
    "keras-cpu": ('h5', ['cpu']),
    "keras-cuda": ('h5', ['cuda']),
    "keras-trt-cpu": ('h5', ['tensorrt', 'cpu']),
    "keras-trt-cuda": ('h5', ['tensorrt', 'cuda']),
    "tflite-cpu": ('tflite', ['cpu']),
    "tflite-cuda": ('tflite', ['cuda']),
    "tflite-trt-cpu": ('tflite', ['tensorrt', 'cpu']),
    "tflite-trt-cuda": ('tflite', ['tensorrt', 'cuda']),
    "pb-cpu": ('pb', ['cpu']),
    "pb-cuda": ('pb', ['cuda']),
    "pb-trt-cpu": ('pb', ['tensorrt', 'cpu']),
    "pb-trt-cuda": ('pb', ['tensorrt', 'cuda']),
    "torch-cpu": ("pth", ["cpu"]),
    "torch-cuda": ("pth", ["cpu"]),
    "torch-trt-cpu": ("pth", ["tensorrt", "cpu"]),
    "torch-trt-cuda": ("pth", ["tensorrt", "cuda"]),
    "onnx-cpu": ('onnx', ['cpu']),
    "onnx-cuda": ('onnx', ['cuda']),
    "onnx-trt-cpu": ('onnx', ['tensorrt', 'cpu']),
    "onnx-trt-cuda": ('onnx', ['tensorrt', 'cuda']),
}


@pytest.mark.parametrize(
    "model_format, composite_target", list(params.values()), ids=list(params.keys())
)
def test_tvm(model_format, composite_target):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        cfg = TVMConfig()
        cfg.cpu_target = "x86-64"
        cfg.composite_target = composite_target
        if model_format == "h5":
            model: tf.keras.Model = tf.keras.applications.mobilenet.MobileNet()
            model.save("tmp.h5")
            input = Model("tmp.h5", spec=get_model_spec("tmp.h5"))
            input.spec.inputs[0].shape = [1, 224, 224, 3]  # type: ignore
            input.spec.outputs[0].shape = [1, 1000]  # type: ignore
            run(input=input, cfg=cfg)
        elif model_format == "tflite":
            model: tf.keras.Model = tf.keras.applications.mobilenet.MobileNet()
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()

            filename = "model.tflite"
            output_path = os.getcwd() + "/" + filename
            with open(output_path, "wb") as w:
                w.write(tflite_model)
            input = Model("model.tflite", spec=get_model_spec("model.tflite"))
            run(input=input, cfg=cfg)
        elif model_format == "pb":
            model: tf.keras.Model = tf.keras.applications.mobilenet.MobileNet()
            wrapper = tf.function(lambda x: model(x))
            wrapper = wrapper.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))  # type: ignore

            # Get frozen ConcreteFunction
            from tensorflow.python.framework.convert_to_constants import (
                convert_variables_to_constants_v2,
            )

            frozen_func = convert_variables_to_constants_v2(wrapper)
            frozen_func.graph.as_graph_def()

            tf.io.write_graph(
                graph_or_graph_def=frozen_func.graph,
                logdir=".",
                name="frozen_graph.pb",
                as_text=False,
            )

            inputs = []
            outputs = []
            for inp in frozen_func.inputs:
                shape = [1 if x is None else x for x in inp.shape]
                inputs.append(
                    TensorSpec(name=inp.name.split(":")[0], shape=shape, dtype=inp.dtype.name)
                )
            for out in frozen_func.outputs:
                shape = [1 if x is None else x for x in out.shape]
                outputs.append(
                    TensorSpec(name=out.name.split(":")[0], shape=shape, dtype=out.dtype.name)
                )
            spec = ModelSpec(inputs=inputs, outputs=outputs)

            input = Model("frozen_graph.pb", spec=spec)
            run(input=input, cfg=cfg)
        elif model_format == "onnx":
            resnet18 = torchvision.models.resnet18(pretrained=True)
            dummy_input = torch.randn(1, 3, 224, 224)
            onnx_model_path = "./resnet18.onnx"
            torch.onnx.export(resnet18, dummy_input, onnx_model_path)
            input = Model("./resnet18.onnx", spec=get_model_spec("resnet18.onnx"))
            run(input=input, cfg=cfg)
        elif model_format == "pth":
            resnet18 = torchvision.models.resnet18(pretrained=True)
            resnet18 = resnet18.eval()
            dummy_input = torch.randn(1, 3, 224, 224)
            scripted_model = torch.jit.trace(resnet18, dummy_input).eval()  # type: ignore
            scripted_model.save("model.pth")
            spec = ModelSpec(
                inputs=[TensorSpec(name="input0", shape=[1, 3, 224, 224], dtype="float32")],
                outputs=[TensorSpec(name="output0", shape=[1, 1000], dtype="float32")],
            )
            input = Model("./model.pth", spec=spec)
            run(input=input, cfg=cfg)
