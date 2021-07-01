import numpy as np
import os
import tempfile
import tarfile
import tensorflow as tf
import tensorflow_datasets as tfds
import tvm
from tvm.driver import tvmc


def benchmark(model, compiled_model_path, hostname, port, target_device):

    if isinstance(model, tf.keras.Model):
        session = tvm.rpc.connect(hostname, port)

        if target_device == "jetson-nano":
            dev = session.cpu()

        with tempfile.TemporaryDirectory() as tmp_dir:
            t = tarfile.open(compiled_model_path)
            t.extractall(tmp_dir)

            graph = open(os.path.join(tmp_dir, "mod.json")).read()
            params = bytearray(open(os.path.join(tmp_dir, "mod.params"), "rb").read())
            session.upload(os.path.join(tmp_dir, "mod.so"))
            lib = session.load_module("mod.so")

        gmodule = tvm.contrib.graph_executor.create(graph, lib, dev)
        gmodule.load_params(params)

        ftimer = gmodule.module.time_evaluator("run", dev)

        # NOTE assume 1 input layer
        input_layer = model.get_layer(index=0)
        config = input_layer.get_config()
        input_shape = tuple([1] + list(config["batch_input_shape"][1:]))

        for _ in range(0, 100):
            input_tensor = np.random.uniform(-1, 1, size=input_shape).astype(config["dtype"])
            gmodule.set_input(0, input_tensor)
            gmodule.run()

        prof_result = ftimer()
        times = prof_result.results

        result = tvmc.TVMCResult(None, times)

        mean_ts = np.mean(result.times) * 1000
        std_ts = np.std(result.times) * 1000
        max_ts = np.max(result.times) * 1000
        min_ts = np.min(result.times) * 1000

        # stat_table = result.format_times()
        # print(f"\n{stat_table}")
        return {"mean": mean_ts, "std": std_ts, "max": max_ts, "min": min_ts}
