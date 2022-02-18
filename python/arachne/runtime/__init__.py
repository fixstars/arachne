import importlib
import tarfile
from logging import getLogger
from typing import Optional

import yaml
from packaging.version import Version

from ..utils.version_utils import (
    get_cuda_version,
    get_cudnn_version,
    get_tensorrt_version,
)
from .module import RuntimeModule
from .module.onnx import ONNXRuntimeModule
from .module.tflite import TFLiteRuntimeModule
from .module.tvm import TVMRuntimeModule

logger = getLogger(__name__)


def validate_environment(env: dict) -> bool:
    valid = True
    for dep in env["dependencies"]:
        if "cuda" in dep:
            cuda_version = get_cuda_version()
            if cuda_version != dep["cuda"]:
                logger.warning(
                    f"The CUDA version:{cuda_version} is not the same as the version specified in env.yaml:{dep['cuda']}"
                )
                valid = False
        if "cudnn" in dep:
            cudnn_version = get_cudnn_version()
            if cudnn_version != dep["cudnn"]:
                logger.warning(
                    f"The cudnn version:{cudnn_version} is not the same as the version specified in env.yaml:{dep['cudnn']}"
                )
                valid = False
        if "tensorrt" in dep:
            tensorrt_version = get_tensorrt_version()
            if tensorrt_version != dep["tensorrt"]:
                logger.warning(
                    f"The tensorrt version:{tensorrt_version} is not the same as the version specified in env.yaml:{dep['tensorrt']}"
                )
                valid = False
        if "pip" in dep:
            for pkg in dep["pip"]:
                for name in pkg.keys():
                    mod = importlib.import_module(name)
                    runtime_version = Version(mod.__version__)
                    dep_version = Version(pkg[name])
                    if (
                        runtime_version.major != dep_version.major
                        or runtime_version.minor != dep_version.minor
                        or runtime_version.micro != dep_version.micro
                    ):
                        logger.warning(
                            f"A python package:{name} version is not the same as the version specified in env.yaml"
                        )
                        valid = False
    return valid


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

        if not validate_environment(env):
            logger.warning("Some environment dependencies are not satisfied")

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
