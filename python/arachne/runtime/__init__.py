import importlib
import tarfile
from typing import Optional

import yaml

from ..utils import get_cuda_version, get_cudnn_version, get_tensorrt_version
from .module import RuntimeModule
from .module.onnx import ONNXRuntimeModule
from .module.tflite import TFLiteRuntimeModule
from .module.tvm import TVMRuntimeModule


def validate_environment(env: dict) -> bool:
    for dep in env["dependencies"]:
        if "cuda" in dep:
            cuda_version = get_cuda_version()
            if cuda_version != dep["cuda"]:
                return False
        if "cudnn" in dep:
            cudnn_version = get_cudnn_version()
            if cudnn_version != dep["cudnn"]:
                return False
        if "tensorrt" in dep:
            tensorrt_version = get_tensorrt_version()
            if tensorrt_version != dep["tensorrt"]:
                return False
        if "pip" in dep:
            for pkg in dep["pip"]:
                for name in pkg.keys():
                    mod = importlib.import_module(name)
                    if mod.__version__ != pkg[name]:
                        return False
    return True


def init(
    package_tar: Optional[str] = None,
    model_file: Optional[str] = None,
    env_file: Optional[str] = None,
    **kwargs,
) -> RuntimeModule:
    assert (
        package_tar is not None or model_file is not None
    ), "package_tar or model_file should not be None"

    if package_tar is not None:
        with tarfile.open(package_tar, "r:gz") as tar:
            for m in tar.getmembers():
                if m.name == "env.yaml":
                    env_file = m.name
                else:
                    model_file = m.name
            tar.extractall(".")

    assert model_file is not None

    env: dict = {}
    if env_file is not None:
        with open(env_file) as file:
            env = yaml.safe_load(file)

        assert validate_environment(env), "invalid runtime environment"

    if model_file.endswith(".tar"):
        return TVMRuntimeModule(
            model=model_file, device_type=env["tvm_device"], model_spec=env["model_spec"]
        )
    elif model_file.endswith(".tflite"):
        return TFLiteRuntimeModule(model=model_file, **kwargs)
    elif model_file.endswith(".onnx"):
        return ONNXRuntimeModule(model=model_file, **kwargs)
    else:
        assert False, f"Unsupported model format ({model_file}) for runtime"
