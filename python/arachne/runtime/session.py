from typing import Optional
from urllib.parse import urlparse

import tvm.rpc
from tvm._ffi.runtime_ctypes import Device as TVMDevice
from tvm.autotvm.measure import request_remote

from arachne.logger import Logger

logger = Logger.logger()


def parse_rpc_tracker_url(rpc_tracker_str):
    rpc_hostname = rpc_port = None

    if rpc_tracker_str:
        parsed_url = urlparse("//%s" % rpc_tracker_str)
        rpc_hostname = parsed_url.hostname
        rpc_port = parsed_url.port or 9090
        logger.info("RPC tracker hostname: %s", rpc_hostname)
        logger.info("RPC tracker port: %s", rpc_port)

    return rpc_hostname, rpc_port


def create_session(rpc_tracker: Optional[str], rpc_key: Optional[str]) -> tvm.rpc.RPCSession:
    hostname, port = parse_rpc_tracker_url(rpc_tracker)

    if hostname:
        # Remote RPC
        if rpc_key:
            logger.debug("running on remote RPC tracker with key %s", rpc_key)
            session = request_remote(rpc_key, hostname, port, timeout=0)
        else:
            logger.debug("running on remote RPC with no key")
            session = tvm.rpc.connect(hostname, port)
    else:
        # Local
        logger.debug("running a local session")
        session = tvm.rpc.LocalSession()

    return session


def create_tvmdev(device: str, session: tvm.rpc.RPCSession) -> TVMDevice:
    logger.debug("device is %s", device)

    return session.device(device)
