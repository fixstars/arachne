import warnings
from abc import ABCMeta

import grpc

from .stubmgr import FileStubManager, ServerStatusStubManager


class RuntimeClientBase(metaclass=ABCMeta):
    def __init__(self, channel: grpc.Channel):
        self.finalized = False
        self.stats_stub_mgr = ServerStatusStubManager(channel)
        self.stats_stub_mgr.trylock()
        self.file_stub_mgr = FileStubManager(channel)

    def finalize(self):
        self.stats_stub_mgr.unlock()
        self.finalized = True

    def __del__(self):
        try:
            if not self.finalized:
                self.finalize()
        except:
            # when server is already shutdown, fail to unlock server.
            warnings.warn(UserWarning("Failed to unlock server"))
