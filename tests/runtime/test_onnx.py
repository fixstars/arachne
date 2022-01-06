import os
import tempfile

import numpy as np
import onnxruntime as ort
import torch
import torchvision

import arachne.runtime
from arachne.data import Model, ModelSpec, TensorSpec
from arachne.tools.torch2onnx import Torch2ONNXConfig, run


def test_tvm_runtime():
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
        output_model = run(input_model, cfg)

        input_data = np.array(np.random.random_sample([1, 3, 224, 224]), dtype=np.float32)  # type: ignore

        # ONNX Runtime
        sess = ort.InferenceSession(output_model.path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        dout = sess.run(output_names=None, input_feed={input_name: input_data})[0]
        del sess

        # Arachne Runtime
        ort_opts = {"providers": ["CPUExecutionProvider"]}
        runtime_module = arachne.runtime.init(model_file=output_model.path, **ort_opts)
        runtime_module.set_input(0, input_data)
        runtime_module.run()
        aout = runtime_module.get_output(0)

        np.testing.assert_equal(actual=aout, desired=dout)

        runtime_module.benchmark()
