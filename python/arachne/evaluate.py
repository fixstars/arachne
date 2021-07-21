import os
import tarfile
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tvm


def evaluate(model, compiled_model_path, ds, hostname, port, target_device):

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

        model.compiled_metrics.reset_state()
        for image, label in tfds.as_numpy(ds):
            input_tensor = image[np.newaxis, :, :, :]
            gmodule.set_input(0, input_tensor)
            gmodule.run()
            pred = gmodule.get_output(0)
            model.compiled_metrics.metrics[0].update_state([label], pred.numpy())
