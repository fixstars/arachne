import platform
import subprocess
import tarfile
import tempfile
from typing import Optional

import onnx
import onnxruntime
import tensorflow as tf
import yaml
from omegaconf import DictConfig, OmegaConf

from .data import Model, ModelSpec, TensorSpec


def get_model_spec(model_file: str) -> Optional[ModelSpec]:
    inputs = []
    outputs = []
    if model_file.endswith(".tflite"):

        interpreter = tf.lite.Interpreter(model_path=model_file)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        for inp in input_details:
            inputs.append(
                TensorSpec(
                    name=inp["name"], shape=inp["shape"].tolist(), dtype=inp["dtype"].__name__
                )
            )
        for out in output_details:
            outputs.append(
                TensorSpec(
                    name=out["name"], shape=out["shape"].tolist(), dtype=out["dtype"].__name__
                )
            )
    elif model_file.endswith(".h5"):
        model = tf.keras.models.load_model(model_file)
        for inp in model.inputs:  # type: ignore
            shape = [-1 if x is None else x for x in inp.shape]
            inputs.append(TensorSpec(name=inp.name, shape=shape, dtype=inp.dtype.name))
        for out in model.outputs:  # type: ignore
            shape = [-1 if x is None else x for x in out.shape]
            outputs.append(TensorSpec(name=out.name, shape=shape, dtype=out.dtype.name))
    elif model_file.endswith("saved_model"):
        model = tf.saved_model.load(model_file)
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
    elif model_file.endswith(".pb"):
        return None
    elif model_file.endswith(".onnx"):

        session = onnxruntime.InferenceSession(model_file)
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
    elif model_file.endswith(".pth"):
        return None

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
    env = {"model_spec": OmegaConf.to_object(model.spec), "dependencies": []}
    if "tvm" in cfg.tools.keys():
        env["tvm_device"] = "cpu"
        targets = list(cfg.tools.tvm.composite_target)
        if "tensorrt" in targets:
            env["dependencies"].append({"tensorrt": get_tensorrt_version()})
        if "cuda" in targets:
            env["dependencies"].append({"cuda": get_cuda_version()})
            env["dependencies"].append({"cudnn": get_cudnn_version()})
            env["tvm_device"] = "cuda"

    pip_deps = []
    if model.file.endswith(".tflite"):
        pip_deps.append({"tensorflow": tf.__version__})
    if model.file.endswith(".onnx"):
        pip_deps.append({"onnx": onnx.__version__})
        pip_deps.append({"onnxruntime": onnxruntime.__version__})
    env["dependencies"].append({"pip": pip_deps})
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(model.file, arcname=model.file.split("/")[-1])

        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(tmp_dir + "/env.yaml", "w") as file:
                yaml.dump(env, file)
                tar.add(tmp_dir + "/env.yaml", arcname="env.yaml")
