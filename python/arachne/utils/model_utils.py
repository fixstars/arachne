import dataclasses
import tarfile
import tempfile
from dataclasses import asdict
from typing import Optional

import onnx
import onnxruntime
import tensorflow as tf
import torch
import tvm
import yaml
from omegaconf import DictConfig, OmegaConf

from ..data import Model, ModelSpec, TensorSpec
from .onnx_utils import get_onnx_model_spec
from .tf_utils import get_keras_model_spec, get_saved_model_spec, get_tflite_model_spec
from .version_utils import (
    get_cuda_version,
    get_cudnn_version,
    get_tensorrt_version,
    get_torch2trt_version,
)


def get_model_spec(model_path: str) -> Optional[ModelSpec]:
    if model_path.endswith(".tflite"):
        return get_tflite_model_spec(model_path)
    elif model_path.endswith(".h5"):
        return get_keras_model_spec(model_path)
    elif model_path.endswith("saved_model"):
        return get_saved_model_spec(model_path)
    elif model_path.endswith(".onnx"):
        return get_onnx_model_spec(model_path)
    elif model_path.endswith(".pb"):
        return None
    elif model_path.endswith(".pth") or model_path.endswith(".pt"):
        return None
    return None


def load_model_spec(spec_file_path: str) -> ModelSpec:
    tmp = OmegaConf.load(spec_file_path)
    tmp = OmegaConf.to_container(tmp)
    assert isinstance(tmp, dict)
    inputs = []
    outputs = []
    for inp in tmp["inputs"]:
        inputs.append(TensorSpec(name=inp["name"], shape=inp["shape"], dtype=inp["dtype"]))
    for out in tmp["outputs"]:
        outputs.append(TensorSpec(name=out["name"], shape=out["shape"], dtype=out["dtype"]))
    return ModelSpec(inputs=inputs, outputs=outputs)


def save_model(model: Model, output_path: str, tvm_cfg: Optional[DictConfig] = None):
    if dataclasses.is_dataclass(model.spec):
        spec = asdict(model.spec)
    else:
        assert False, f"model.spec should be arachne.data.ModelSpec: {model.spec}"
    env = {"model_spec": spec, "dependencies": []}

    pip_deps = []
    if model.path.endswith(".tar"):
        pip_deps.append({"tvm": tvm.__version__})

        assert tvm_cfg is not None, "when save a tvm_package.tar, tvm_cfg must be avaiable"
        env["tvm_device"] = "cpu"

        targets = list(tvm_cfg.composite_target)
        if "tensorrt" in targets:
            env["dependencies"].append({"tensorrt": get_tensorrt_version()})
        if "cuda" in targets:
            env["dependencies"].append({"cuda": get_cuda_version()})
            env["dependencies"].append({"cudnn": get_cudnn_version()})
            env["tvm_device"] = "cuda"

    if model.path.endswith(".tflite"):
        pip_deps.append({"tensorflow": tf.__version__})
    if model.path.endswith("saved_model"):
        pip_deps.append({"tensorflow": tf.__version__})
    if model.path.endswith(".onnx"):
        pip_deps.append({"onnx": onnx.__version__})
        pip_deps.append({"onnxruntime": onnxruntime.__version__})
    if model.path.endswith(".pth"):
        pip_deps.append({"torch": torch.__version__})  # type: ignore
    if model.path.endswith("_trt.pth"):
        pip_deps.append({"torch2trt": get_torch2trt_version()})
    env["dependencies"].append({"pip": pip_deps})
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(model.path, arcname=model.path.split("/")[-1])

        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(tmp_dir + "/env.yaml", "w") as file:
                yaml.dump(env, file)
                tar.add(tmp_dir + "/env.yaml", arcname="env.yaml")
