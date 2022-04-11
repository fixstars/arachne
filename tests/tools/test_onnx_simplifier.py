import os
import subprocess
import sys
import tempfile

import torch
import torch.cuda
import torch.onnx
import torchvision

from arachne.tools.onnx_simplifier import ONNXSimplifier, ONNXSimplifierConfig
from arachne.utils.model_utils import init_from_file


def test_onnx_simplifier():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        dummy_input = torch.randn(1, 3, 224, 224)
        model = torchvision.models.resnet18(pretrained=True)
        onnx_model_file = "resnet18.onnx"
        torch.onnx.export(model, dummy_input, onnx_model_file)

        input_model = init_from_file(onnx_model_file)
        check_n = 10  # number of check iteration
        cfg = ONNXSimplifierConfig(check_n=check_n)
        # The validation of the simplified model is performed in onnx-simplifier.
        ONNXSimplifier.run(input_model, cfg)


def test_cli():

    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        dummy_input = torch.randn(1, 3, 224, 224)
        model = torchvision.models.resnet18(pretrained=True)
        onnx_model_file = "resnet18.onnx"
        torch.onnx.export(model, dummy_input, onnx_model_file)

        check_n = 10  # number of check iteration
        ret = subprocess.run(
            [
                sys.executable,
                "-m",
                "arachne.driver.cli",
                "+tools=onnx_simplifier",
                f"model_file={onnx_model_file}",
                "output_path=simplified.onnx",
                f"tools.onnx_simplifier.check_n={check_n}",
            ]
        )

        assert ret.returncode == 0
