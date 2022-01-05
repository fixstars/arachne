import dataclasses
import platform
import subprocess
import tarfile
import tempfile
from dataclasses import asdict
from typing import Callable, Dict, Optional

import onnx
import onnxruntime
import tensorflow as tf
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from tvm.relay import Any

from .data import Model, ModelSpec, TensorSpec


def get_torch_dtype_from_string(dtype_str: str) -> torch.dtype:
    dtype_str_to_torch_dtype_dict = {
        "bool": torch.bool,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
    if dtype_str not in dtype_str_to_torch_dtype_dict:
        assert False, f"Not conversion map for {dtype_str}"
    return dtype_str_to_torch_dtype_dict[dtype_str]


def get_tflite_model_spec(model_path: str) -> ModelSpec:
    inputs = []
    outputs = []
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for inp in input_details:
        inputs.append(
            TensorSpec(name=inp["name"], shape=inp["shape"].tolist(), dtype=inp["dtype"].__name__)
        )
    for out in output_details:
        outputs.append(
            TensorSpec(name=out["name"], shape=out["shape"].tolist(), dtype=out["dtype"].__name__)
        )
    return ModelSpec(inputs=inputs, outputs=outputs)


def get_keras_model_spec(model_path: str) -> ModelSpec:
    inputs = []
    outputs = []
    model = tf.keras.models.load_model(model_path)
    for inp in model.inputs:  # type: ignore
        shape = [-1 if x is None else x for x in inp.shape]
        inputs.append(TensorSpec(name=inp.name, shape=shape, dtype=inp.dtype.name))
    for out in model.outputs:  # type: ignore
        shape = [-1 if x is None else x for x in out.shape]
        outputs.append(TensorSpec(name=out.name, shape=shape, dtype=out.dtype.name))
    return ModelSpec(inputs=inputs, outputs=outputs)


def get_saved_model_spec(model_path: str) -> ModelSpec:
    inputs = []
    outputs = []
    model = tf.saved_model.load(model_path)
    model_inputs = [
        inp for inp in model.signatures["serving_default"].inputs if "unknown" not in inp.name  # type: ignore
    ]
    model_outputs = [
        out for out in model.signatures["serving_default"].outputs if "unknown" not in out.name  # type: ignore
    ]
    for inp in model_inputs:
        shape = [-1 if x is None else x for x in inp.shape]
        inputs.append(TensorSpec(name=inp.name, shape=shape, dtype=inp.dtype.name))
    for out in model_outputs:
        shape = [-1 if x is None else x for x in out.shape]
        outputs.append(TensorSpec(name=out.name, shape=shape, dtype=out.dtype.name))
    return ModelSpec(inputs=inputs, outputs=outputs)


def get_onnx_model_spec(model_path: str) -> ModelSpec:
    inputs = []
    outputs = []

    session = onnxruntime.InferenceSession(model_path)
    for inp in session.get_inputs():
        dtype = inp.type.replace("tensor(", "").replace(")", "")
        if dtype == "float":
            dtype = "float32"
        elif dtype == "double":
            dtype = "float64"
        inputs.append(TensorSpec(name=inp.name, shape=inp.shape, dtype=dtype))
    for out in session.get_outputs():
        dtype = out.type.replace("tensor(", "").replace(")", "")
        if dtype == "float":
            dtype = "float32"
        elif dtype == "double":
            dtype = "float64"
        outputs.append(TensorSpec(name=out.name, shape=out.shape, dtype=dtype))
    return ModelSpec(inputs=inputs, outputs=outputs)


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


def get_tensorrt_version():
    dist = platform.linux_distribution()[0]
    if dist == "Ubuntu" or dist == "Debian":
        result = subprocess.check_output("dpkg -l | grep libnvinfer-dev", shell=True)
        return result.decode().strip().split()[2]
    else:
        # TODO: Support Fedora (RedHat)
        assert False, "Unsupported OS distribution"


def get_cuda_version():
    result = subprocess.check_output("nvcc --version", shell=True)
    return result.decode().strip().split("\n")[-1].replace(",", "").split()[-2]


def get_cudnn_version():
    dist = platform.linux_distribution()[0]
    if dist == "Ubuntu" or dist == "Debian":
        result = subprocess.check_output("dpkg -l | grep libcudnn", shell=True)
        return result.decode().strip().split()[2]
    else:
        # TODO: Support Fedora (RedHat)
        assert False, "Unsupported OS distribution"


def save_model(model: Model, output_path: str, cfg: DictConfig):
    if dataclasses.is_dataclass(model.spec):
        spec = asdict(model.spec)
    else:
        assert False, f"model.spec is unknown object or None: {model.spec}"
    env = {"model_spec": spec, "dependencies": []}
    if "tvm" in cfg.tools.keys():
        targets = list(cfg.tools.tvm.composite_target)
        if len(targets) > 0:
            env["tvm_device"] = "cpu"
        if "tensorrt" in targets:
            env["dependencies"].append({"tensorrt": get_tensorrt_version()})
        if "cuda" in targets:
            env["dependencies"].append({"cuda": get_cuda_version()})
            env["dependencies"].append({"cudnn": get_cudnn_version()})
            env["tvm_device"] = "cuda"

    pip_deps = []
    if model.path.endswith(".tflite"):
        pip_deps.append({"tensorflow": tf.__version__})
    if model.path.endswith("saved_model"):
        pip_deps.append({"tensorflow": tf.__version__})
    if model.path.endswith(".onnx"):
        pip_deps.append({"onnx": onnx.__version__})
        pip_deps.append({"onnxruntime": onnxruntime.__version__})
    env["dependencies"].append({"pip": pip_deps})
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(model.path, arcname=model.path.split("/")[-1])

        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(tmp_dir + "/env.yaml", "w") as file:
                yaml.dump(env, file)
                tar.add(tmp_dir + "/env.yaml", arcname="env.yaml")


_TOOL_CONFIG_GLOBAL_OBJECTS: Dict[str, Any] = {}


def get_tool_config_objects():
    return _TOOL_CONFIG_GLOBAL_OBJECTS


_TOOL_RUN_GLOBAL_OBJECTS: Dict[str, Callable] = {}


def get_tool_run_objects():
    return _TOOL_RUN_GLOBAL_OBJECTS
