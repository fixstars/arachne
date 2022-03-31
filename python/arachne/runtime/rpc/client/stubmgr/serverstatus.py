import grpc

from arachne.runtime.rpc.protobuf import server_status_pb2, server_status_pb2_grpc


class ServerStatusStubManager:
    def __init__(self, channel: grpc.Channel):
        """Check if the gRPC server is available using ServerStatusStubManager.

        Only one client per server can be connected at the same time.

        Args:
            channel (grpc.Channel): channel to connect server.
        """
        self.channel = channel
        self.stub = server_status_pb2_grpc.ServerStatusStub(channel)
        self.lock_success = False

    def trylock(self):
        """Try to lock server.

        Raises:
            grpc.RpcError: raises when server failes to lock.
        """
        try:
            self.stub.Lock(server_status_pb2.LockRequest())
        except grpc.RpcError:
            raise
        self.lock_success = True

    def unlock(self):
        """Unlock server."""
        if self.lock_success:
            self.stub.Unlock(server_status_pb2.UnlockRequest())
