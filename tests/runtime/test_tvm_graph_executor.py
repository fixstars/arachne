import os
import tempfile

import numpy as np
import tensorflow as tf
import tvm
from omegaconf import OmegaConf
from tvm.contrib import graph_executor
from tvm.contrib.graph_executor import GraphModule

import arachne.runtime
from arachne.data import Model
from arachne.runtime.module.tvm import _open_module_file
from arachne.tools.tvm import TVMConfig, run
from arachne.utils.model_utils import get_model_spec, save_model


def test_tvm_runtime():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)

        model = tf.keras.applications.mobilenet.MobileNet()
        model.save("tmp.h5")

        cfg = TVMConfig()
        cfg.cpu_target = "x86-64"
        cfg.composite_target = ["cpu"]
        input = Model("tmp.h5", spec=get_model_spec("tmp.h5"))
        input.spec.inputs[0].shape = [1, 224, 224, 3]  # type: ignore
        input.spec.outputs[0].shape = [1, 1000]  # type: ignore
        output = run(input=input, cfg=cfg)
        save_model(output, "package.tar", tvm_cfg=OmegaConf.structured(cfg))

        input_data = np.array(np.random.random_sample([1, 224, 224, 3]), dtype=np.float32)  # type: ignore

        # TVM Graph Executor
        tvm_device = tvm.runtime.device("cpu", 0)
        graph, params, lib = _open_module_file(output.path)
        module: GraphModule = graph_executor.create(graph, lib, tvm_device)
        module.load_params(params)
        module.set_input(0, input_data)
        module.run()
        dout = module.get_output(0).numpy()
        del module

        # Arachne Runtime
        runtime_module = arachne.runtime.init(package_tar="package.tar")
        runtime_module.set_input(0, input_data)
        runtime_module.run()
        aout = runtime_module.get_output(0)

        np.testing.assert_equal(actual=aout, desired=dout)

        runtime_module.benchmark()
