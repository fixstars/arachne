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
    env_file: Optional[str] = None,
    rpc_host: str = "localhost",
    rpc_port: int = 5051,
    **kwargs,
) -> RuntimeClientBase:
    """Initialize RuntimeClient to send requests to the server.

    The arguments to be passed as model file are different for runtime:

    - ONNX/TfLite:   set :code:`model_file`
    - TVM: set :code:`package_tar` or set both :code:`model_file` and :code:`env_file`

    Args:
        package_tar (Optional[str], optional): TVM package filepath archived by arachne.tools.tvm. Defaults to None.
        model_file (Optional[str], optional): ONNX/TfLite/TVM model filepath. Defaults to None.
        env_file (Optional[str], optional): environment file :code:`env.yaml`. Defaults to None.
        rpc_host (str, optional): host name of gRPC server. Defaults to "localhost".
        rpc_port (int, optional): port number of gRPC server. Defaults to 5051.

    Returns:
        RuntimeClientBase: ONNX/TfLite/TVM RuntimeClient
    """

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
        if package_tar is None:
            assert env_file is not None
            package_tar = "./package.tar"
            with tarfile.open(package_tar, "w:gz") as tar:
                tar.add(model_file, arcname=model_file.split("/")[-1])
                tar.add(env_file, arcname="env.yaml")
        return TVMRuntimeClient(channel, package_tar, **kwargs)
    elif model_file.endswith(".tflite"):
        return TfliteRuntimeClient(channel, model_file, **kwargs)
    elif model_file.endswith(".onnx"):
        return ONNXRuntimeClient(channel, model_file, **kwargs)
    else:
        assert False, f"Unsupported model format ({model_file}) for runtime"
