import argparse
from concurrent import futures

import grpc

from arachne.runtime.rpc.logger import Logger

from .protobuf import fileserver_pb2_grpc, runtime_pb2_grpc, server_status_pb2_grpc
from .servicer import FileServicer, RuntimeServicer, ServerStatusServicer

logger = Logger.logger()


def create_channel(host: str = "localhost", port: int = 5051) -> grpc.Channel:
    """Create gRPC channel.

    Args:
        host (str, optional): Defaults to "localhost".
        port (int, optional): Defaults to 5051.

    Returns:
        grpc.Channel:
    """
    rpc_address = f"{host}:{port}"
    channel = grpc.insecure_channel(rpc_address)
    return channel


def create_server(port: int):
    """Create the server by specifying the runtime.

    Args:
        runtime_name (str): The servicer corresponding to the specified runtime is registered with the server.
        port (int):

    Returns:
        grpc.Server
    """
    server = grpc.server(
        thread_pool=futures.ThreadPoolExecutor(max_workers=1), options=(("grpc.so_reuseport", 0),)
    )

    server_status_pb2_grpc.add_ServerStatusServicer_to_server(ServerStatusServicer(), server)
    fileserver_pb2_grpc.add_FileServiceServicer_to_server(FileServicer(), server)
    runtime_pb2_grpc.add_RuntimeServicer_to_server(RuntimeServicer(), server)

    server.add_insecure_port("[::]:" + str(port))
    return server


def start_server(server: grpc.Server, port: int):
    """Start server and wait for termination."""
    server.start()
    logger.info(f"server is running on port: {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5051)

    args = parser.parse_args()

    server = create_server(args.port)
    start_server(server, args.port)
