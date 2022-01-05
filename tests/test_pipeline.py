import os
import tempfile

import pytest
import tensorflow as tf
import torch
import torchvision
from omegaconf import OmegaConf

from arachne.data import Model, ModelSpec, TensorSpec
from arachne.pipeline import PipelineConfig, get_default_tool_configs, run
from arachne.utils import get_model_spec


@pytest.mark.parametrize("pipeline", [["tflite_converter", "tvm"], ["tftrt"]])
def test_pipeline_from_keras(pipeline):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        model = tf.keras.applications.mobilenet.MobileNet()
        model_path = "tmp-saved_model"
        model.save(model_path)
        input = Model(path=model_path, spec=get_model_spec(model_path))

        cfg = PipelineConfig()
        cfg.pipeline = pipeline
        cfg.tools = get_default_tool_configs(pipeline)
        cfg = OmegaConf.structured(cfg)
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

        input = Model(path="resnet18.pt", spec=spec)

        cfg = PipelineConfig()
        cfg.pipeline = pipeline
        cfg.tools = get_default_tool_configs(pipeline)
        cfg = OmegaConf.structured(cfg)
        run(input, cfg)
