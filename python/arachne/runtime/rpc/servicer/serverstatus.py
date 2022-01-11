from arachne.runtime.rpc.protobuf import server_status_pb2_grpc
from arachne.runtime.rpc.protobuf.msg_response_pb2 import MsgResponse


class ServerStatusServicer(server_status_pb2_grpc.ServerStatusServicer):
    def __init__(self):
        self.is_busy = False

    def Lock(self, request, context):
        if self.is_busy:
            return MsgResponse(error=True, message="server is busy")
        else:
            print("lock!")
            self.is_busy = True
            return MsgResponse(error=False, message="OK")

    def Unlock(self, request, context):
        if self.is_busy:
            print("server is unlocked.")
            self.is_busy = False
        return MsgResponse(error=False, message="OK")
