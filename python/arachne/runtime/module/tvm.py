import os
import tarfile
import tempfile
import time
from typing import Dict, Optional, Tuple

import numpy as np
import tvm
import tvm.rpc
from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor
from tvm.contrib.graph_executor import GraphModule
from tvm.runtime.module import Module as TVMModule

from . import RuntimeModule


def _open_module_file(file: str) -> Tuple[Optional[str], Optional[bytearray], TVMModule]:
    with tempfile.TemporaryDirectory() as tmp_dir:
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
        lib = tvm.runtime.load_module(tmp_dir + "/mod.tar")

    return graph, params, lib


class TVMRuntimeModule(RuntimeModule):
    def __init__(self, model: str, device_type: str, model_spec: dict, **kwargs):
        tvm_device = tvm.runtime.device(device_type, 0)
        graph, params, lib = _open_module_file(model)
        if "TVM_DEBUG_EXECUTOR" in os.environ:
            module: GraphModule = debug_executor.create(
                graph, lib, tvm_device, dump_root="./tvm_dbg"
            )
        else:
            module: GraphModule = graph_executor.create(graph, lib, tvm_device)
        module.load_params(params)

        self.input_details = self.output_details = {}
        if "inputs" in model_spec and "outputs" in model_spec:
            self.input_details = model_spec["inputs"]
            self.output_details = model_spec["outputs"]

        if self.input_details == {} or self.output_details == {}:
            print("input_details and output_details are not set")

        self.module: GraphModule = module
        self.tvm_device = tvm_device

    def run(self):
        self.module.run()

    def set_input(self, idx, value, **kwargs):
        self.module.set_input(idx, value, **kwargs)

    def get_output(self, idx):
        return self.module.get_output(idx)

    def get_input_details(self):
        return self.input_details

    def get_output_details(self):
        return self.output_details

    def benchmark(self, warmup: int = 1, repeat: int = 10, number: int = 1) -> Dict:
        for idx, inp in enumerate(self.input_details):
            input_data = np.random.uniform(0, 1, size=inp["shape"]).astype(inp["dtype"])
            self.set_input(idx, input_data)

        for _ in range(warmup):
            self.run()

        times = []
        for _ in range(repeat):
            time_start = time.perf_counter()
            for _ in range(number):
                self.run()
            time_end = time.perf_counter()
            times.append((time_end - time_start) / number)

        mean_ts = np.mean(times) * 1000
        std_ts = np.std(times) * 1000
        max_ts = np.max(times) * 1000
        min_ts = np.min(times) * 1000

        return {"mean": mean_ts, "std": std_ts, "max": max_ts, "min": min_ts}
