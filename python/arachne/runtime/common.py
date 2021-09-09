from typing import Optional

import tvm.rpc

from arachne.pipeline.package import Package
from arachne.runtime.module.registry import get_module_class
from arachne.runtime.session import create_session
from arachne.types import IndexedOrderedDict

from .module.module import RuntimeModule


def create_runtime(package: Package, session: tvm.rpc.RPCSession, profile: bool) -> RuntimeModule:
    runtime_module_class = get_module_class(package)

    if runtime_module_class is None:
        raise RuntimeError(f"This package ({package.__class__.__name__}) cannot run.")

    return runtime_module_class(package, session, profile)


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
