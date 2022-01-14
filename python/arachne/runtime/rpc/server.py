import argparse
from concurrent import futures

import grpc

from arachne.logger import Logger

from .protobuf import fileserver_pb2_grpc, server_status_pb2_grpc
from .servicer import (
    FileServicer,
    ServerStatusServicer,
    get_runtime_servicer,
    runtime_servicer_list,
)

logger = Logger.logger()


def create_channel(host: str = "localhost", port: int = 5051) -> grpc.Channel:
    rpc_address = f"{host}:{port}"
    channel = grpc.insecure_channel(rpc_address)
    return channel


def create_server(runtime_name: str, port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    server_status_pb2_grpc.add_ServerStatusServicer_to_server(ServerStatusServicer(), server)
    fileserver_pb2_grpc.add_FileServiceServicer_to_server(FileServicer(), server)

    servicer_class = get_runtime_servicer(runtime_name)
    assert servicer_class is not None
    servicer_class.register_servicer_to_server(server)

    server.add_insecure_port("[::]:" + str(port))
    return server


def start_server(server: grpc.Server, port: int):
    server.start()
    logger.info(f"server is running on port: {port}")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5051)
    parser.add_argument("--runtime", type=str, choices=runtime_servicer_list())

    args = parser.parse_args()

    server = create_server(args.runtime, args.port)
    start_server(server, args.port)
