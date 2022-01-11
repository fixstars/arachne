from arachne.runtime.rpc.protobuf import server_status_pb2, server_status_pb2_grpc


class ServerStatusClient:
    def __init__(self, channel):
        self.channel = channel
        self.stub = server_status_pb2_grpc.ServerStatusStub(channel)
        self.lock_success = False

    def trylock(self):
        response = self.stub.Lock(server_status_pb2.LockRequest())
        if response.error:
            raise Exception(response.message)
        else:
            self.lock_success = True

    def unlock(self):
        if self.lock_success:
            return self.stub.Unlock(server_status_pb2.UnlockRequest())
