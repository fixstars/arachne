import os
import tempfile

import numpy as np
import tensorflow as tf
import torch
import torchvision

from arachne.data import Model, ModelSpec, TensorSpec
from arachne.tools.openvino2tf import OpenVINO2TFConfig
from arachne.tools.openvino2tf import run as run_openvino2tf
from arachne.tools.openvino_mo import OpenVINOModelOptConfig
from arachne.tools.openvino_mo import run as run_openvino_mo
from arachne.tools.torch2onnx import Torch2ONNXConfig
from arachne.tools.torch2onnx import run as run_torch2onnx


def test_torch2onnx():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        resnet18 = torchvision.models.resnet18(pretrained=True).eval()
        torch.save(resnet18, f="resnet18.pt")

        spec = ModelSpec(
            inputs=[TensorSpec(name="input0", shape=[1, 3, 224, 224], dtype="float32")],
            outputs=[TensorSpec(name="output0", shape=[1, 1000], dtype="float32")],
        )
        input_model = Model(file="resnet18.pt", spec=spec)
        cfg = Torch2ONNXConfig()
        m = run_torch2onnx(input_model, cfg)
        m = run_openvino_mo(m, OpenVINOModelOptConfig())
        m = run_openvino2tf(m, OpenVINO2TFConfig())

        tf_loaded = tf.saved_model.load(m.file)
        resnet18_tf = tf_loaded.signatures["serving_default"]  # type: ignore

        input = np.random.rand(1, 3, 224, 224).astype(np.float32)  # type: ignore
        torch_input = torch.from_numpy(input).clone()
        tf_input = tf.convert_to_tensor(np.transpose(input, (0, 2, 3, 1)))

        torch_result = resnet18(torch_input).to('cpu').detach().numpy().copy()
        tf_result = resnet18_tf(tf_input)
        tf_result = tf_result['tf.identity'].numpy()

        np.testing.assert_allclose(torch_result, tf_result, atol=1e-5, rtol=1e-5)
