import os
import tempfile

import pytest
import tensorflow as tf
import torch
import torchvision

from arachne.data import Model, ModelFormat, ModelSpec, TensorSpec
from arachne.driver.pipeline import PipelineConfig, get_default_tool_configs, run
from arachne.utils.model_utils import init_from_dir


@pytest.mark.parametrize("pipeline", [["tflite_converter", "tvm"], ["tftrt"]])
def test_pipeline_from_keras(pipeline):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        model = tf.keras.applications.mobilenet.MobileNet()
        model_path = "tmp-saved_model"
        model.save(model_path)

        input = init_from_dir(model_path)

        cfg = PipelineConfig()
        cfg.pipeline = pipeline
        cfg.tools = get_default_tool_configs(pipeline)
        run(input, cfg)


@pytest.mark.parametrize("pipeline", [["torch2onnx", "openvino_mo", "openvino2tf"], ["torch2trt"]])
def test_pipeline_from_torch(pipeline):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)

        resnet18 = torchvision.models.resnet18(pretrained=True)
        torch.save(resnet18, f="resnet18.pt")

        spec = ModelSpec(
            inputs=[TensorSpec(name="input0", shape=[1, 3, 224, 224], dtype="float32")],
            outputs=[TensorSpec(name="output0", shape=[1, 1000], dtype="float32")],
        )

        input = Model(path="resnet18.pt", format=ModelFormat.PYTORCH, spec=spec)

        cfg = PipelineConfig()
        cfg.pipeline = pipeline
        cfg.tools = get_default_tool_configs(pipeline)
        run(input, cfg)
