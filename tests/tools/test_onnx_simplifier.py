import os
import subprocess
import sys
import tempfile

import torch
import torch.cuda
import torchvision

from arachne.data import Model
from arachne.tools.onnx_simplifier import OnnxSimplifierConfig, run
from arachne.utils.model_utils import get_model_spec


def test_onnx_simplifier():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        dummy_input = torch.randn(1, 3, 224, 224)
        model = torchvision.models.resnet18(pretrained=True)
        onnx_model_file = "resnet18.onnx"
        torch.onnx.export(model, dummy_input, onnx_model_file)

        input_model = Model(path=onnx_model_file, spec=get_model_spec(onnx_model_file))
        check_n = 10  # number of check iteration
        cfg = OnnxSimplifierConfig(cli_args=str(check_n))
        # The validation of the simplified model is performed in onnx-simplifier.
        run(input_model, cfg)


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
                "arachne.tools.onnx_simplifier",
                f"input={onnx_model_file}",
                "output=simplified.onnx",
                f"tools.onnx_simplifier.cli_args={check_n}",
            ]
        )

        assert ret.returncode == 0
