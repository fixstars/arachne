import time

import numpy as np

from .factory import RuntimeModuleBase, RuntimeModuleFactory


@RuntimeModuleFactory.register("tflite")
class TFLiteRuntimeModule(RuntimeModuleBase):
    def __init__(self, model: str, **kwargs):
        import tensorflow as tf

        self.module: tf.lite.Interpreter = tf.lite.Interpreter(model_path=model, **kwargs)
        self.module.allocate_tensors()
        self.input_details = self.module.get_input_details()
        self.output_details = self.module.get_output_details()

    def run(self):
        self.module.invoke()

    def set_input(self, idx, value, **kwargs):
        """Set input data.

        Args:
            idx (int): layer index to set data
            np_arr (np.ndarray): input data
        """
        if not isinstance(idx, int):
            raise Exception("idx should be int")
        self.module.set_tensor(self.input_details[idx]["index"], value)

    def get_output(self, idx):
        """Get inference output.

        Args:
            index (int): layer index to get output

        Returns:
            np.ndarray: output data
        """
        return self.module.get_tensor(self.output_details[idx]["index"])

    def get_input_details(self):
        return self.input_details

    def get_output_details(self):
        return self.output_details

    def benchmark(self, warmup: int = 1, repeat: int = 10, number: int = 1):
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
