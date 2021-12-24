import os
import tempfile

import torch
import torchvision

from arachne.data import Model, ModelSpec, TensorSpec
from arachne.tools.torch2onnx import Torch2ONNXConfig, run


def test_torch2onnx():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        resnet18 = torchvision.models.resnet18(pretrained=True)
        torch.save(resnet18, f="resnet18.pt")

        spec = ModelSpec(
            inputs=[TensorSpec(name="input0", shape=[1, 3, 224, 224], dtype="float32")],
            outputs=[TensorSpec(name="output0", shape=[1, 1000], dtype="float32")],
        )
        input_model = Model(path="resnet18.pt", spec=spec)
        cfg = Torch2ONNXConfig()
        run(input_model, cfg)
