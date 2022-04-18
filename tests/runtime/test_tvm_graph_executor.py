import os
import tarfile
import tempfile

import numpy as np
import tvm
from tvm.contrib import graph_executor
from tvm.contrib.download import download
from tvm.contrib.graph_executor import GraphModule

import arachne.runtime
from arachne.runtime.module.tvm import _open_module_file


def test_tvm_runtime():
    with tempfile.TemporaryDirectory() as tmp_dir:
        url = "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/tvm_mobilenet.tar"

        tvm_package_path = tmp_dir + "/tvm_mobilenet.tar"
        download(url, tvm_package_path)

        os.chdir(tmp_dir)
        with tarfile.open(tvm_package_path, "r:gz") as tar:
            tar.extractall(".")
        tvm_model_path = tmp_dir + "/tvm_package_0.tar"

        input_data = np.array(np.random.random_sample([1, 224, 224, 3]), dtype=np.float32)  # type: ignore

        # TVM Graph Executor
        tvm_device = tvm.runtime.device("cpu", 0)
        graph, params, lib = _open_module_file(tvm_model_path)
        module: GraphModule = graph_executor.create(graph, lib, tvm_device)
        module.load_params(params)
        module.set_input(0, input_data)
        module.run()
        dout = module.get_output(0).numpy()
        del module

        # Arachne Runtime
        runtime_module = arachne.runtime.init(runtime="tvm", package_tar=tvm_package_path)
        runtime_module.set_input(0, input_data)
        runtime_module.run()
        aout = runtime_module.get_output(0)

        np.testing.assert_equal(actual=aout, desired=dout)

        runtime_module.benchmark()
