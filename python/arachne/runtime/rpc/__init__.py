import tarfile
from typing import Optional

from .client import (
    ONNXRuntimeClient,
    RuntimeClientBase,
    TfliteRuntimeClient,
    TVMRuntimeClient,
)
from .server import create_channel, create_server, start_server
from .servicer import (
    FileServicer,
    ONNXRuntimeServicer,
    ServerStatusServicer,
    TfLiteRuntimeServicer,
    TVMRuntimeServicer,
)


def init(
    package_tar: Optional[str] = None,
    model_file: Optional[str] = None,
    rpc_host: str = "localhost",
    rpc_port: int = 5051,
    **kwargs,
) -> RuntimeClientBase:

    assert (
        package_tar is not None or model_file is not None
    ), "package_tar or model_file should not be None"

    if package_tar is not None:
        with tarfile.open(package_tar, "r:gz") as tar:
            for m in tar.getmembers():
                if m.name != "env.yaml":
                    model_file = m.name
            tar.extractall(".")

    assert model_file is not None

    channel = create_channel(rpc_host, rpc_port)
    if model_file.endswith(".tar"):
        assert package_tar is not None
        return TVMRuntimeClient(channel, package_tar, **kwargs)
    elif model_file.endswith(".tflite"):
        return TfliteRuntimeClient(channel, model_file, **kwargs)
    elif model_file.endswith(".onnx"):
        return ONNXRuntimeClient(channel, model_file, **kwargs)
    else:
        assert False, f"Unsupported model format ({model_file}) for runtime"
