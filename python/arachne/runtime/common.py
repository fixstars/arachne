import os
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import tvm.rpc
from tvm._ffi.runtime_ctypes import Device as TVMDevice
from tvm.autotvm.measure import request_remote
from tvm.contrib import graph_executor, tflite_runtime
from tvm.contrib.debugger import debug_executor
from tvm.runtime.profiler_vm import VirtualMachineProfiler
from tvm.runtime.vm import VirtualMachine

from arachne.logger import Logger
from arachne.pipeline.package import Package, TFLitePackage, TVMPackage, TVMVMPackage
from arachne.runtime.session import create_session
from arachne.types import IndexedOrderedDict

from .module.module import RuntimeModule
from .module.tflite import TFLiteRuntimeModule
from .module.tvm import TVMRuntimeModule, TVMVMRuntimeModule


def create_runtime(package: Package, session: tvm.rpc.RPCSession, profile: bool) -> RuntimeModule:
    if isinstance(package, TFLitePackage):
        return TFLiteRuntimeModule(package, session, profile)
    elif isinstance(package, TVMPackage):
        return TVMRuntimeModule(package, session, profile)
    elif isinstance(package, TVMVMPackage):
        return TVMVMRuntimeModule(package, session, profile)
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
