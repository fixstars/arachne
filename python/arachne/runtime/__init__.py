import importlib
import tarfile
from typing import Optional

import tensorflow as tf
import yaml

from .module import RuntimeModule
from .module.tflite import TFLiteRuntimeModule
from .module.tvm import TVMRuntimeModule


def validate_environment(env: dict) -> bool:
    sys_details = tf.sysconfig.get_build_info()
    for dep in env["dependencies"]:
        if "cuda" in dep:
            cuda_version = sys_details["cuda_version"]
            if cuda_version != dep["cuda"]:
                return False
        if "cudnn" in dep:
            cudnn_version = sys_details["cudnn_version"]
            if cudnn_version != dep["cudnn"]:
                return False
        if "pip" in dep:
            for pkg in dep["pip"]:
                for name in pkg.keys():
                    mod = importlib.import_module(name)
                    if mod.__version__ != pkg[name]:
                        return False
    return True


def init(
    package: Optional[str] = None, model: Optional[str] = None, env_file: Optional[str] = None
) -> Optional[RuntimeModule]:
    assert package is not None or model is not None, "package or model should not be None"

    if package is not None:
        with tarfile.open(package, "r:gz") as tar:
            for m in tar.getmembers():
                if m.name == "env.yaml":
                    env_file = m.name
                else:
                    model = m.name
            tar.extractall(".")

    assert model is not None

    env: dict = {}
    if env_file is not None:
        with open(env_file) as file:
            env = yaml.safe_load(file)

        assert validate_environment(env), "invalid runtime environment"
    if model.endswith(".tar"):
        return TVMRuntimeModule(
            model=model, device_type=env["tvm_device"], model_spec=env["model_spec"]
        )
    elif model.endswith(".tflite"):
        return TFLiteRuntimeModule(model=model)
    else:
        assert False, f"Unsupported model format ({model}) for runtime"
