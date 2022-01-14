import os
import tempfile

import numpy as np
import pytest
import torch
import torch.cuda
import torchvision
from torch2trt import TRTModule

from arachne.data import Model, ModelSpec, TensorSpec
from arachne.tools.torch2trt import Torch2TRTConfig, run


def create_dummy_representative_dataset():
    datasets = []
    shape = [1, 3, 224, 224]
    dtype = "float32"
    for _ in range(100):
        datasets.append(np.random.rand(*shape).astype(np.dtype(dtype)))  # type: ignore

    np.save("dummy.npy", datasets)


def check_torch2trt_output(torch_model, input_shape, precision, torch_trt_model_path):
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)  # type: ignore
    torch_input = torch.from_numpy(input_data).clone()
    torch_model.eval()
    dout = torch_model(torch_input).to("cpu").detach().numpy().copy()

    model_trt = TRTModule()
    torch.cuda.empty_cache()
    torch_input = torch_input.to("cuda")
    model_trt.load_state_dict(torch.load(torch_trt_model_path))
    model_trt = model_trt.to("cuda")
    aout = model_trt(torch_input)
    aout = aout.to("cpu").detach().numpy().copy()

    if precision == "FP32":
        np.testing.assert_allclose(aout, dout, atol=1e-5, rtol=1e-5)
    elif precision == "FP16":
        np.testing.assert_allclose(aout, dout, atol=0.1, rtol=0)


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
        input_model = Model(path="resnet18.pt", spec=spec)
        cfg = Torch2TRTConfig()
        if precision == "FP16":
            cfg.fp16_mode = True
        output = run(input_model, cfg)
        check_torch2trt_output(resnet18, [1, 3, 224, 224], precision, output.path)


@pytest.mark.parametrize(
    "calib_algo", ["ENTROPY_CALIBRATION_2", "ENTROPY_CALIBRATION", "MINMAX_CALIBRATION"]
)
def test_torch2trt_int8(calib_algo):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        resnet18 = torchvision.models.resnet18(pretrained=True)
        torch.save(resnet18, f="resnet18.pt")

        spec = ModelSpec(
            inputs=[TensorSpec(name="input0", shape=[1, 3, 224, 224], dtype="float32")],
            outputs=[TensorSpec(name="output0", shape=[1, 1000], dtype="float32")],
        )
        input_model = Model(path="resnet18.pt", spec=spec)
        cfg = Torch2TRTConfig()
        cfg.int8_mode = True
        cfg.int8_calib_algorithm = calib_algo
        create_dummy_representative_dataset()
        cfg.int8_calib_dataset = "dummy.npy"
        run(input_model, cfg)
