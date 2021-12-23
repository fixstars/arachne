import os
import tempfile

import pytest
import torch
import torchvision

from arachne.data import Model, ModelSpec, TensorSpec
from arachne.tools.torch2trt import Torch2TRTConfig, run


@pytest.mark.parametrize("precision", ["FP32", "FP16"])
def test_torch2trt(precision):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        resnet18 = torchvision.models.resnet18(pretrained=True)
        torch.save(resnet18, f="resnet18.pt")

        spec = ModelSpec(
            inputs=[TensorSpec(name="input0", shape=[1, 3, 224, 224], dtype="float32")],
            outputs=[TensorSpec(name="output0", shape=[1, 1000], dtype="float32")],
        )
        input_model = Model(file="resnet18.pt", spec=spec)
        cfg = Torch2TRTConfig()
        if precision == "FP16":
            cfg.fp16_mode = True
        run(input_model, cfg)


# @pytest.mark.parametrize("calib_algo", ["ENTROPY_CALIBRATION_2", "ENTROPY_CALIBRATION", "LEGACY_CALIBRATION", "MINMAX_CALIBRATION"])
@pytest.mark.parametrize("calib_algo", ["ENTROPY_CALIBRATION_2", "ENTROPY_CALIBRATION", "MINMAX_CALIBRATION"])
def test_torch2trt_int8(calib_algo):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        resnet18 = torchvision.models.resnet18(pretrained=True)
        torch.save(resnet18, f="resnet18.pt")

        spec = ModelSpec(
            inputs=[TensorSpec(name="input0", shape=[1, 3, 224, 224], dtype="float32")],
            outputs=[TensorSpec(name="output0", shape=[1, 1000], dtype="float32")],
        )
        input_model = Model(file="resnet18.pt", spec=spec)
        cfg = Torch2TRTConfig()
        cfg.int8_mode = True
        cfg.int8_calib_algorithm = calib_algo
        run(input_model, cfg)
