import grpc

from .client import (
    FileClient,
    ONNXRuntimeClient,
    ServerStatusClient,
    TfliteRuntimeClient,
    TVMRuntimeClient,
)
from .servicer import (
    FileServicer,
    ONNXRuntimeServicer,
    ServerStatusServicer,
    TfLiteRuntimeServicer,
    TVMRuntimeServicer,
)


def create_channel(host: str = "localhost", port: int = 5051) -> grpc.Channel:
    rpc_address = f"{host}:{port}"
    channel = grpc.insecure_channel(rpc_address)
    return channel
