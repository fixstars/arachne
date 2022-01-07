import os
import tempfile

import numpy as np
import onnxruntime as ort
import torch
import torchvision

from arachne.data import Model, ModelSpec, TensorSpec
from arachne.tools.torch2onnx import Torch2ONNXConfig, run


def check_torch2onnx_output(torch_model, input_shape, onnx_model_path):
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)  # type: ignore
    torch_model.eval()
    torch_input = torch.from_numpy(input_data).clone()
    dout = torch_model(torch_input).to("cpu").detach().numpy().copy()

    sess = ort.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name
    aout = sess.run(output_names=None, input_feed={input_name: input_data})[0]
    np.testing.assert_allclose(aout, dout, atol=1e-5, rtol=1e-5)


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
        output = run(input_model, cfg)
        check_torch2onnx_output(resnet18, [1, 3, 224, 224], output.path)
