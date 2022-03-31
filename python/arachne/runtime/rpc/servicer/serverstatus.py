import grpc

from arachne.runtime.rpc.logger import Logger
from arachne.runtime.rpc.protobuf import server_status_pb2_grpc
from arachne.runtime.rpc.protobuf.msg_response_pb2 import MsgResponse

logger = Logger.logger()


class ServerStatusServicer(server_status_pb2_grpc.ServerStatusServicer):
    """ServerStatusServicer prevents multiple clients to connect.

    Only one client per server can be connected at the same time.
    """

    def __init__(self):
        self.is_busy = False

    def Lock(self, request, context):
        """
        Lock server. If it fails, set an error to response context.

        Returns:
            MsgResponse:
        """
        if self.is_busy:
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details("server is being used by other client")
            return MsgResponse()
        else:
            logger.info("server is locked from client")
            self.is_busy = True
            return MsgResponse(msg="locked")

    def Unlock(self, request, context):
        """
        Unlock server.

        Returns:
            MsgResponse:
        """
        if self.is_busy:
            logger.info("server is unlocked from client")
            self.is_busy = False
            return MsgResponse(msg="unlocked")
        else:
            return MsgResponse(msg="already unlocked")
