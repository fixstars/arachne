import os
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from tvm.autotvm.measure import request_remote
from tvm.contrib import graph_executor, tflite_runtime
from tvm.contrib.debugger import debug_executor
import tvm.driver.tvmc.common as tvmccommon
import tvm.rpc
from tvm.runtime.module import Module as TVMModule
from tvm._ffi.runtime_ctypes import Device as TVMDevice

from .device import Device
from .logger import Logger
from .module import RuntimeModule
from .types import IndexedOrderedDict


logger = Logger.logger()


def create_session(rpc_tracker: str, rpc_key: str) -> tvm.rpc.RPCSession:
    hostname, port = tvmccommon.tracker_host_port_from_cli(rpc_tracker)

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

    # TODO(Maruoka): Support more devices
    if device == "cuda":
        tvmdev = session.cuda()
    elif device == "cl":
        tvmdev = session.cl()
    else:
        assert device == "cpu"
        tvmdev = session.cpu()

    return tvmdev


def open_module_file(
    file: Path,
    session: tvm.rpc.RPCSession,
    device: Device
) -> Tuple[str, bytearray, TVMModule]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.debug("extracting module file %s", file)
        t = tarfile.open(file)
        t.extractall(tmp_dir)
        graph = open(os.path.join(tmp_dir, "mod.json")).read()
        params = bytearray(
            open(os.path.join(tmp_dir, "mod.params"), "rb").read())

        session.upload(os.path.join(tmp_dir, "mod.so"))
        lib = session.load_module("mod.so")

    return graph, params, lib


def create_runtime(
    file: Path,
    session: tvm.rpc.RPCSession,
    device: Device,
    tvmdev: TVMDevice,
    profile: bool
) -> RuntimeModule:
    if device.is_tflite:
        with open(file, "rb") as model_fin:
            module = tflite_runtime.create(model_fin.read(), tvmdev)
    elif device.is_edgetpu:
        with open(file, "rb") as model_fin:
            module = tflite_runtime.create(
                model_fin.read(), tvmdev, runtime_target="edge_tpu")
    else:
        graph, params, lib = open_module_file(file, session, device)

        if profile:
            logger.debug("creating runtime with profiling enabled")
            # TODO(Maruoka): Set dump_root into under '.artifacts/{experiment}'
            module = debug_executor.create(
                graph, lib, tvmdev, dump_root="./.prof")
        else:
            logger.debug("creating runtime with profiling disabled")
            module = graph_executor.create(graph, lib, tvmdev)

        logger.debug("load params into the runtime module")
        module.load_params(params)

    return RuntimeModule(module)


def runner_init(
    module_file: Path,
    device: Device,
    rpc_tracker: Optional[str] = None,
    rpc_key: Optional[str] = None,
    profile: bool = False
) -> Tuple[RuntimeModule, TVMDevice]:
    session = create_session(rpc_tracker, rpc_key)

    tvmdev = create_tvmdev(device.tvmdev, session)

    module = create_runtime(module_file, session, device, tvmdev, profile)

    return module, tvmdev


def run_module(
    module: RuntimeModule,
    tvmdev: TVMDevice,
    inputs: IndexedOrderedDict,
    output_info: IndexedOrderedDict
) -> IndexedOrderedDict:
    module.set_inputs(inputs, tvmdev)
    module.run()
    return module.get_outputs(output_info)
