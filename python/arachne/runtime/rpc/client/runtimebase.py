import warnings

import grpc

from .serverstatus import ServerStatusClient


class RuntimeClientBase:
    def __init__(self, channel: grpc.Channel):
        self.finalized = False
        self.statusclient = ServerStatusClient(channel)
        self.statusclient.trylock()

    def finalize(self):
        self.statusclient.unlock()
        self.finalized = True

    def __del__(self):
        try:
            if not self.finalized:
                self.finalize()
        except:
            warnings.warn(UserWarning("Failed to unlock server."))
