import time
from concurrent import futures

import grpc

from arachne.runtime.rpc.protobuf import (
    fileserver_pb2_grpc,
    tfliteruntime_pb2_grpc,
    tvmruntime_pb2_grpc,
)
from arachne.runtime.rpc.servicer import (
    FileServicer,
    TfLiteRuntimeServicer,
    TVMRuntimeServicer,
)


def create_server(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    tfliteruntime_pb2_grpc.add_TfliteRuntimeServerServicer_to_server(
        TfLiteRuntimeServicer(), server
    )
    fileserver_pb2_grpc.add_FileServerServicer_to_server(FileServicer(), server)
    tvmruntime_pb2_grpc.add_TVMRuntimeServerServicer_to_server(TVMRuntimeServicer(), server)

    server.add_insecure_port("[::]:" + str(port))
    return server


def start_server(server: grpc.Server, port: int):
    server.start()
    print("run server on port", port)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        server.stop(0)


import sys

if __name__ == "__main__":
    port = int(sys.argv[1])
    server = create_server(port)
    start_server(server, port)
