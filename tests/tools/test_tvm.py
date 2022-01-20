import os
import tempfile
from typing import List

import numpy as np
import onnxruntime as ort
import pytest
import tensorflow as tf
import torch
import torch.jit
import torch.onnx
import torchvision
import tvm
from tvm.contrib import graph_executor
from tvm.contrib.download import download
from tvm.contrib.graph_executor import GraphModule

from arachne.data import Model, ModelSpec, TensorSpec
from arachne.runtime.module.tvm import _open_module_file
from arachne.tools.tvm import TVMConfig, get_predefined_config, run
from arachne.utils.model_utils import get_model_spec
from arachne.utils.tf_utils import make_tf_gpu_usage_growth

params = {
    "keras-cpu": ("h5", ["cpu"]),
    "keras-cuda": ("h5", ["cuda"]),
    "keras-trt-cpu": ("h5", ["tensorrt", "cpu"]),
    "keras-trt-cuda": ("h5", ["tensorrt", "cuda"]),
    "tflite-cpu": ("tflite", ["cpu"]),
    "tflite-cuda": ("tflite", ["cuda"]),
    "tflite-trt-cpu": ("tflite", ["tensorrt", "cpu"]),
    "tflite-trt-cuda": ("tflite", ["tensorrt", "cuda"]),
    "pb-cpu": ("pb", ["cpu"]),
    "pb-cuda": ("pb", ["cuda"]),
    "pb-trt-cpu": ("pb", ["tensorrt", "cpu"]),
    "pb-trt-cuda": ("pb", ["tensorrt", "cuda"]),
    "torch-cpu": ("pth", ["cpu"]),
    "torch-cuda": ("pth", ["cpu"]),
    "torch-trt-cpu": ("pth", ["tensorrt", "cpu"]),
    "torch-trt-cuda": ("pth", ["tensorrt", "cuda"]),
    "onnx-cpu": ("onnx", ["cpu"]),
    "onnx-cuda": ("onnx", ["cuda"]),
    "onnx-trt-cpu": ("onnx", ["tensorrt", "cpu"]),
    "onnx-trt-cuda": ("onnx", ["tensorrt", "cuda"]),
}


def check_tvm_output(
    input_model: str, input_fmt: str, input_shape: List[int], tvm_model: str, tvm_device_type: str
):
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)  # type: ignore
    if input_fmt == "h5" or input_fmt == "pb":
        model = tf.keras.models.load_model(input_model)
        dout = model(input_data).numpy()  # type: ignore
    elif input_fmt == "tflite":
        interpreter = tf.lite.Interpreter(model_path=input_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        dout = interpreter.get_tensor(output_details[0]["index"])
    elif input_fmt == "onnx":
        sess = ort.InferenceSession(input_model)
        input_name = sess.get_inputs()[0].name
        dout = sess.run(output_names=None, input_feed={input_name: input_data})[0]
    elif input_fmt == "pth":
        model = torch.load(input_model)
        model.eval()
        torch_input = torch.from_numpy(input_data).clone()
        dout = model(torch_input).to("cpu").detach().numpy().copy()
    else:
        assert False

    tvm_device = tvm.runtime.device(tvm_device_type, 0)
    graph, params, lib = _open_module_file(tvm_model)
    module: GraphModule = graph_executor.create(graph, lib, tvm_device)
    module.load_params(params)
    module.set_input(0, input_data)
    module.run()
    aout = module.get_output(0).numpy()
    del module

    np.testing.assert_allclose(aout, dout, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "model_format, composite_target", list(params.values()), ids=list(params.keys())
)
def test_tvm(model_format, composite_target):

    make_tf_gpu_usage_growth()

    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        cfg = TVMConfig()
        cfg.cpu_target = "x86-64"
        cfg.composite_target = composite_target
        device_type = "cuda" if "cuda" in composite_target else "cpu"
        if model_format == "h5":
            model = tf.keras.applications.mobilenet.MobileNet()
            model.save("tmp.h5")
            input = Model("tmp.h5", spec=get_model_spec("tmp.h5"))
            input.spec.inputs[0].shape = [1, 224, 224, 3]  # type: ignore
            input.spec.outputs[0].shape = [1, 1000]  # type: ignore
            output = run(input=input, cfg=cfg)

            check_tvm_output("tmp.h5", model_format, [1, 224, 224, 3], output.path, device_type)

        elif model_format == "tflite":
            url = (
                "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/mobilenet.tflite"
            )

            model_path = "mobilenet.tflite"
            download(url, model_path)
            input = Model(model_path, spec=get_model_spec(model_path))
            output = run(input=input, cfg=cfg)

            check_tvm_output(
                "mobilenet.tflite", model_format, [1, 224, 224, 3], output.path, device_type
            )

        elif model_format == "pb":
            model = tf.keras.applications.mobilenet.MobileNet()
            model.save("tmp.h5")
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
            output = run(input=input, cfg=cfg)

            check_tvm_output("tmp.h5", model_format, [1, 224, 224, 3], output.path, device_type)

        elif model_format == "onnx":
            url = (
                "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/resnet18.onnx"
            )

            model_path = "resnet18.onnx"
            download(url, model_path)
            input = Model(model_path, spec=get_model_spec(model_path))
            output = run(input=input, cfg=cfg)

            check_tvm_output(
                "resnet18.onnx", model_format, [1, 3, 224, 224], output.path, device_type
            )

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
            output = run(input=input, cfg=cfg)

            check_tvm_output("model.pth", model_format, [1, 3, 224, 224], output.path, device_type)


@pytest.mark.parametrize("target", ["dgx-1", "dgx-s", "jetson-nano", "jetson-xavier-nx", "rasp4b64"])
def test_predefined_config(target):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)

        cfg = get_predefined_config(target)

        model = tf.keras.applications.mobilenet.MobileNet()
        model.save("tmp.h5")
        input = Model("tmp.h5", spec=get_model_spec("tmp.h5"))
        input.spec.inputs[0].shape = [1, 224, 224, 3]  # type: ignore
        input.spec.outputs[0].shape = [1, 1000]  # type: ignore
        run(input=input, cfg=cfg)
