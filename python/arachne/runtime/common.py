import os
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import tvm.driver.tvmc.common as tvmccommon
import tvm.rpc
from tvm._ffi.runtime_ctypes import Device as TVMDevice
from tvm.autotvm.measure import request_remote
from tvm.contrib import graph_executor, tflite_runtime
from tvm.contrib.debugger import debug_executor
from tvm.driver.tvmc.common import parse_target
from tvm.runtime.module import Module as TVMModule
from tvm.runtime.vm import VirtualMachine
from arachne.logger import Logger
from arachne.pipeline.package import Package, TFLitePackage, TVMPackage
from arachne.pipeline.package.tvm_vm import TVMVMPackage
from arachne.types import IndexedOrderedDict

from .module import RuntimeModule, TFLiteRuntimeModule, TVMRuntimeModule, TVMVMRuntimeModule

logger = Logger.logger()


def create_session(rpc_tracker: Optional[str], rpc_key: Optional[str]) -> tvm.rpc.RPCSession:
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

    return session.device(device)


def open_module_file(file: Path, session: tvm.rpc.RPCSession) -> Tuple[Optional[str], Optional[bytearray], TVMModule]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.debug("extracting module file %s", file)
        with tarfile.open(file) as t:
            t.extractall(tmp_dir)
        graph = None
        params = None
        graph_path = os.path.join(tmp_dir, "mod.json")
        if os.path.exists(graph_path):
            graph = open(graph_path).read()
        params_path = os.path.join(tmp_dir, "mod.params")
        if os.path.exists(params_path):
            params = bytearray(open(params_path, "rb").read())
        session.upload(os.path.join(tmp_dir, "mod.tar"))
        lib = session.load_module("mod.tar")

    return graph, params, lib


def create_runtime(package: Package, session: tvm.rpc.RPCSession, profile: bool) -> RuntimeModule:
    if isinstance(package, TFLitePackage):
        tvmdev = create_tvmdev("cpu", session)
        runtime_target = "edge_tpu" if package.for_edgetpu else "cpu"
        with open(package.dir / package.model_file, "rb") as model_fin:
            module = tflite_runtime.create(model_fin.read(), tvmdev, runtime_target)
        return TFLiteRuntimeModule(module, tvmdev, package)
    elif isinstance(package, (TVMPackage, TVMVMPackage)):
        targets = parse_target(package.target)
        target = targets[-1]["raw"]
        tvmdev = create_tvmdev(target, session)
        graph, params, lib = open_module_file(package.dir / package.package_file, session)

        if isinstance(package, TVMPackage):
            if profile:
                logger.debug("creating runtime with profiling enabled")
                # TODO(Maruoka): Set dump_root into under '.artifacts/{experiment}'
                module = debug_executor.create(graph, lib, tvmdev, dump_root="./.prof")
            else:
                logger.debug("creating runtime with profiling disabled")
                module = graph_executor.create(graph, lib, tvmdev)

            logger.debug("load params into the runtime module")
            module.load_params(params)
            return TVMRuntimeModule(module, tvmdev, package)
        else:
            module = VirtualMachine(lib, tvmdev)
            return TVMVMRuntimeModule(module, tvmdev, package)
    else:
        raise RuntimeError(f"This package ({package.__class__.__name__}) cannot run.")


def runner_init(
    package: Package,
    rpc_tracker: Optional[str] = None,
    rpc_key: Optional[str] = None,
    profile: bool = False,
) -> RuntimeModule:
    session = create_session(rpc_tracker, rpc_key)

    module = create_runtime(package, session, profile)

    return module


def run_module(
    module: RuntimeModule,
    inputs: IndexedOrderedDict,
    output_info: IndexedOrderedDict,
) -> IndexedOrderedDict:
    module.set_inputs(inputs)
    module.run()
    return module.get_outputs(output_info)
