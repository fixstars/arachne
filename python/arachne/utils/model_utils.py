import dataclasses
import os
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

from ..data import Model, ModelFormat, ModelSpec, TensorSpec
from .onnx_utils import get_onnx_model_spec
from .tf_utils import get_keras_model_spec, get_saved_model_spec, get_tflite_model_spec
from .version_utils import (
    get_cuda_version,
    get_cudnn_version,
    get_tensorrt_version,
    get_torch2trt_version,
)


def init_from_file(model_file: str) -> Model:
    """The function to initialize arachne.data.Model from a model file

    Args:
        model_file (str):  path to a model file

    Returns:
        Model: a model instance
    """
    format: ModelFormat
    spec: Optional[ModelSpec]
    if model_file.endswith(".tflite"):
        format = ModelFormat.TFLITE
        spec = get_tflite_model_spec(model_file)
    elif model_file.endswith(".h5"):
        format = ModelFormat.KERAS_H5
        spec = get_keras_model_spec(model_file)
    elif model_file.endswith(".onnx"):
        format = ModelFormat.ONNX
        spec = get_onnx_model_spec(model_file)
    elif model_file.endswith(".pb"):
        format = ModelFormat.TF_PB
        spec = None
    elif model_file.endswith(".pth") or model_file.endswith(".pt"):
        format = ModelFormat.PYTORCH
        spec = None
    else:
        raise RuntimeError("Fail to detect a model format for " + model_file)

    return Model(path=model_file, format=format, spec=spec)


def __is_saved_model_dir(model_dir: str):
    found_pb = False
    found_assets = False
    found_variables = False
    for f in os.listdir(model_dir):
        if f.endswith(".pb"):
            found_pb = True
        if f == "assets":
            found_assets = True
        if f == "variables":
            found_variables = True
    return found_pb & found_assets & found_variables


def __is_openvino_model_dir(model_dir: str):
    found_bin = False
    found_xml = False
    found_mapping = False

    for f in os.listdir(model_dir):
        if f.endswith(".bin"):
            found_bin = True
        if f.endswith(".xml"):
            found_xml = True
        if f.endswith(".mapping"):
            found_mapping = True
    return found_bin & found_xml & found_mapping


def __is_caffe_model_dir(model_dir: str):
    found_caffemodel = False
    found_prototxt = False

    for f in os.listdir(model_dir):
        if f.endswith(".caffemodel"):
            found_caffemodel = True
        if f.endswith(".prototxt"):
            found_prototxt = True
    return found_caffemodel & found_prototxt


def init_from_dir(model_dir: str) -> Model:
    """The function to initialize arachne.data.Model from a model directory

    Args:
        model_dir (str):  path to a model directory

    Returns:
        Model: a model instance
    """
    format: ModelFormat
    spec: Optional[ModelSpec]

    if __is_saved_model_dir(model_dir):
        format = ModelFormat.TF_SAVED_MODEL
        spec = get_saved_model_spec(model_dir)
    elif __is_openvino_model_dir(model_dir):
        format = ModelFormat.OPENVINO
        spec = None
    elif __is_caffe_model_dir(model_dir):
        format = ModelFormat.CAFFE
        spec = None
    else:
        raise RuntimeError("Fail to detect a model format for " + model_dir)

    return Model(path=model_dir, format=format, spec=spec)


def load_model_spec(spec_file_path: str) -> ModelSpec:
    """The function to load the model specification from a YAML file

    Args:
        spec_file_path (str):  path to a YAML file that describes the model specification

    Returns:
        ModelSpec: the tensor information of the model or None
    """
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
    """The function to save the model that is a tool output as a TAR file

    Args:
        model (Model):  a tool output model
        output_path (str): an output path
        tvm_cfg (:obj:`DictConfig`, optional): pass to the TVM config if the model depends on the TVM

    """
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
