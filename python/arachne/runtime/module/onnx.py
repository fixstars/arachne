import time
from typing import List

import numpy as np
import onnxruntime as ort

from . import RuntimeModule


def onnx_tensor_type_to_np_dtype(ottype: str):
    dtype = ottype.replace("tensor(", "").replace(")", "")
    if dtype == "float":
        dtype = "float32"
    elif dtype == "double":
        dtype = "float64"
    return dtype


class ONNXRuntimeModule(RuntimeModule):
    def __init__(
        self, model: str, provider_options: List[str] = ["CPUExecutionProvider"], **kwargs
    ):
        self.module: ort.InferenceSession = ort.InferenceSession(model, **kwargs)
        self._inputs = {}
        self._outputs = {}
        self.input_details = self.module.get_inputs()
        self.output_details = self.module.get_outputs()

    def run(self):
        # NOTE: should we support run_options?
        self._outputs = self.module.run(output_names=None, input_feed=self._inputs)

    def set_input(self, idx, value, **kwargs):
        self._inputs[self.input_details[idx].name] = value

    def get_output(self, idx):
        return self._outputs[idx]

    def get_input_details(self):
        return self.input_details

    def get_output_details(self):
        return self.output_details

    def benchmark(self, warmup: int = 1, repeat: int = 10, number: int = 1):
        for idx, inp in enumerate(self.input_details):
            input_data = np.random.uniform(0, 1, size=inp.shape).astype(
                onnx_tensor_type_to_np_dtype(inp.type)
            )
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
