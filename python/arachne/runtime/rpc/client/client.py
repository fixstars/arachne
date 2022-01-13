import warnings
from abc import ABCMeta

import grpc

from .serverstatus import ServerStatusClient


class RuntimeClientBase(metaclass=ABCMeta):
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
            # when server is already shutdown, fail to unlock server.
            warnings.warn(UserWarning("Failed to unlock server"))
