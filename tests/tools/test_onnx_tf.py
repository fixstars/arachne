import os
import subprocess
import sys
import tarfile
import tempfile

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import torch
import torch.cuda
import torch.onnx
import torchvision

from arachne.data import Model
from arachne.tools.onnx_tf import ONNXTf, ONNXTfConfig
from arachne.utils.model_utils import get_model_spec


def check_onnx_tf_output(onnx_model_path, input_shape, tf_file_path):
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)  # type: ignore
    sess = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dout = sess.run(output_names=None, input_feed={input_name: input_data})[0]

    infer = tf.saved_model.load(tf_file_path).signatures["serving_default"]  # type: ignore
    result_dic = infer(tf.constant(input_data))
    aout = list(result_dic.values())[0]

    np.testing.assert_allclose(aout, dout, atol=1e-5, rtol=1e-5)


def test_onnx_tf():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        dummy_input = torch.randn(1, 3, 224, 224)
        model = torchvision.models.resnet18(pretrained=True)
        onnx_model_file = "resnet18.onnx"
        torch.onnx.export(model, dummy_input, onnx_model_file)

        input_model = Model(path=onnx_model_file, spec=get_model_spec(onnx_model_file))
        cfg = ONNXTfConfig()
        output = ONNXTf.run(input_model, cfg)
        check_onnx_tf_output(onnx_model_file, [1, 3, 224, 224], output.path)


def test_cli():

    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        dummy_input = torch.randn(1, 3, 224, 224)
        model = torchvision.models.resnet18(pretrained=True)
        onnx_model_file = "resnet18.onnx"
        torch.onnx.export(model, dummy_input, onnx_model_file)

        ret = subprocess.run(
            [
                sys.executable,
                "-m",
                "arachne.driver.cli",
                "+tools=onnx_tf",
                f"input={onnx_model_file}",
                "output=output.tar",
            ]
        )
        assert ret.returncode == 0
        saved_model_dir = None
        with tarfile.open("output.tar", "r:gz") as tar:
            saved_model_dir = tar.getmembers()[0].name
            tar.extractall(".")
        check_onnx_tf_output(onnx_model_file, [1, 3, 224, 224], saved_model_dir)
