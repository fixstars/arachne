import os
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tvm
import tvm.rpc
from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor
from tvm.contrib.graph_executor import GraphModule
from tvm.runtime.module import Module as TVMModule
from tvm.runtime.profiler_vm import VirtualMachineProfiler
from tvm.runtime.vm import VirtualMachine

from arachne.logger import Logger
from arachne.runtime.indexed_ordered_dict import IndexedOrderedDict
from arachne.runtime.package import TVMPackage, TVMVMPackage
from arachne.runtime.session import create_tvmdev

from ._registry import register_module_class
from .module import RuntimeModule

logger = Logger.logger()


def open_module_file(
    file: Path, session: tvm.rpc.RPCSession
) -> Tuple[Optional[str], Optional[bytearray], TVMModule]:
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


class TVMRuntimeModule(RuntimeModule):
    """
    A wrapper class for tvm.contrib.graph_executor.GraphModule
    """

    def __init__(self, package: TVMPackage, session: tvm.rpc.RPCSession, profile: bool):
        assert isinstance(package, TVMPackage)
        target = package.target_tvmdev
        tvmdev = create_tvmdev(target, session)
        graph, params, lib = open_module_file(package.dir / package.package_file, session)

        if profile:
            logger.debug("creating runtime with profiling enabled")
            # TODO(Maruoka): Set dump_root into under '.artifacts/{experiment}'
            module = debug_executor.create(graph, lib, tvmdev, dump_root="./.prof")
        else:
            logger.debug("creating runtime with profiling disabled")
            module = graph_executor.create(graph, lib, tvmdev)

            logger.debug("load params into the runtime module")
            module.load_params(params)

        self.module: GraphModule = module
        self.tvmdev = tvmdev
        self.package = package

    def get_name(self):
        return "tvm_runtime_module"

    def set_inputs(self, inputs: Union[IndexedOrderedDict, List]):
        if isinstance(inputs, IndexedOrderedDict):
            self.module.set_input(**inputs)
        elif isinstance(inputs, list):
            for i, input in enumerate(inputs):
                self.module.set_input(key=i, value=input)
        else:
            raise RuntimeError("unreachable")

    def run(self):
        self.module.run()

    def benchmark(self, repeat: int) -> Dict:
        input_tensors = [
            np.random.uniform(-1, 1, size=ispec.shape).astype(ispec.dtype)
            for ispec in self.package.input_info.values()
        ]

        for i, tensor in enumerate(input_tensors):
            self.set_input(i, tensor)

        timer = self.module.module.time_evaluator("run", self.tvmdev, 1, repeat=repeat)

        self.run()

        prof_result = timer()
        times = prof_result.results

        mean_ts = np.mean(times) * 1000
        std_ts = np.std(times) * 1000
        max_ts = np.max(times) * 1000
        min_ts = np.min(times) * 1000

        return {"mean": mean_ts, "std": std_ts, "max": max_ts, "min": min_ts}

    def get_outputs(self, output_info: IndexedOrderedDict) -> IndexedOrderedDict:
        outputs: IndexedOrderedDict = IndexedOrderedDict()
        for i, name in enumerate(output_info.keys()):
            outputs[name] = self.module.get_output(i).asnumpy()

        return outputs

    def set_input(self, key=None, value=None, **params):
        """A wrapper for GraphModule.set_input()

        Parameters
        ----------
        key : int or str
           The input idx/name

        value : the input value.

        params: additional arguments
        """
        self.module.set_input(key, value, **params)

    def get_input(self, idx: int, out=None):
        """A wrapper for GraphModule.get_input()

        Parameters
        ----------
        idx : int
           The input idx

        out : NDArray or None
            The output array container

        Returns
        -------
        NDArray : the input tensor value
        """
        return self.module.get_input(idx, out)

    def get_num_inputs(self):
        """A wrapper for GraphModule.get_num_inputs()"""
        return self.module.get_num_inputs()

    def get_output(self, idx: int, out=None):
        """A wrapper for GraphModule.get_output()

        Parameters
        ----------
        index : int
            The output index

        out : NDArray
            The output array container

        Returns
        -------
        NDArray : the output tensor value
        """
        return self.module.get_output(idx, out)

    def get_num_outputs(self):
        """A wrapper for GraphModule.get_num_outputs()"""
        return self.module.get_num_outputs()


register_module_class(TVMPackage, TVMRuntimeModule)


class TVMVMRuntimeModule(RuntimeModule):
    def __init__(self, package: TVMVMPackage, session: tvm.rpc.RPCSession, profile: bool):
        assert isinstance(package, TVMVMPackage)
        target = package.target_tvmdev
        tvmdev = create_tvmdev(target, session)
        graph, params, lib = open_module_file(package.dir / package.package_file, session)

        if profile:
            logger.debug("creating runtime with profiling enabled")
            module = VirtualMachineProfiler(lib, tvmdev)
        else:
            logger.debug("creating runtime with profiling disabled")
            module = VirtualMachine(lib, tvmdev)

        self.module: GraphModule = module
        self.tvmdev = tvmdev
        self.package = package

    def get_name(self) -> str:
        return "tvm_vm_runtime_module"

    def set_inputs(self, inputs: Union[IndexedOrderedDict, List]):
        if isinstance(inputs, IndexedOrderedDict):
            assert len(self.package.input_info.keys()) == len(inputs.keys())
            tvm_input_dict = {}
            for k, v in inputs.items():
                tvm_input_dict[k] = tvm.nd.array(v, self.tvmdev)
            self.module.set_input("main", **tvm_input_dict)
        elif isinstance(inputs, list):
            assert len(inputs) == len(self.package.input_info.keys())
            tvm_inputs = []
            for input in inputs:
                tvm_inputs.append(tvm.nd.array(input, self.tvmdev))
            self.module.set_input("main", *tvm_inputs)
        else:
            raise RuntimeError("unreachable")

    def run(self):
        if isinstance(self.module, VirtualMachineProfiler):
            vmprofile_res = self.module.profile(func_name="main")
            print(vmprofile_res)
        elif isinstance(self.module, VirtualMachine):
            self.module.invoke_stateful("main")
        else:
            raise Exception("unreachable")

    def benchmark(self, repeat: int) -> Dict:
        input_tensors = [
            np.random.uniform(0, 255, size=ispec.shape).astype(ispec.dtype)
            for ispec in self.package.input_info.values()
        ]

        self.set_inputs(input_tensors)

        timer = self.module.module.time_evaluator("invoke_stateful", self.tvmdev, 1, repeat=repeat)
        self.run()

        prof_result = timer("main")
        times = prof_result.results

        # because first execution is slow, calculate average time from the second execution
        first_ts = times[0] * 1000
        mean_ts = np.mean(times[1:]) * 1000
        std_ts = np.std(times[1:]) * 1000
        max_ts = np.max(times[1:]) * 1000
        min_ts = np.min(times[1:]) * 1000

        return {
            "first": first_ts,
            "mean_rest": mean_ts,
            "std_rest": std_ts,
            "max_rest": max_ts,
            "min_rest": min_ts,
        }

    def get_outputs(self, output_info: IndexedOrderedDict) -> IndexedOrderedDict:
        outputs: IndexedOrderedDict = IndexedOrderedDict()
        module_outputs = self.__vmobj_to_list(self.module.get_outputs())
        for i, name in enumerate(self.package.output_info.keys()):
            if name in output_info.keys():
                outputs[name] = module_outputs[i][0]

        return outputs

    def __vmobj_to_list(self, output):
        if isinstance(output, tvm.nd.NDArray):
            return [output.asnumpy()]
        elif isinstance(output, tvm.runtime.container.ADT) or isinstance(output, list):
            return [self.__vmobj_to_list(f) for f in output]
        else:
            raise RuntimeError("Unknown object type: %s" % type(output))


register_module_class(TVMVMPackage, TVMVMRuntimeModule)
