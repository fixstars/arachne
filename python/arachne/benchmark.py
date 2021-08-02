import os
import tarfile
import tempfile
from typing import List, Optional

import numpy as np
import tvm
from tvm.driver import tvmc

from . import common, device
from .ishape import InputSpec


def benchmark_tvm_model(
    compiled_model_path: str,
    input_specs: List[InputSpec],
    hostname: Optional[str],
    rpc_key: Optional[str],
    target_device: str,
    profile: bool,
):
    session = common.create_session(hostname, rpc_key)

    dev = device.get_device(target_device)
    tvmdev = common.create_tvmdev(dev.tvmdev, session)

    with tempfile.TemporaryDirectory() as tmp_dir:
        t = tarfile.open(compiled_model_path)
        t.extractall(tmp_dir)

        graph = open(os.path.join(tmp_dir, "mod.json")).read()
        params = bytearray(open(os.path.join(tmp_dir, "mod.params"), "rb").read())
        session.upload(os.path.join(tmp_dir, "mod.tar"))
        lib = session.load_module("mod.tar")

    if profile:
        gmodule = tvm.contrib.debugger.debug_executor.create(graph, lib, tvmdev)
    else:
        gmodule = tvm.contrib.graph_executor.create(graph, lib, tvmdev)
    gmodule.load_params(params)

    input_tensors = [
        np.random.uniform(-1, 1, size=ispec.shape).astype(ispec.dtype) for ispec in input_specs
    ]

    for i, tensor in enumerate(input_tensors):
        gmodule.set_input(i, tensor)

    timer = gmodule.module.time_evaluator("run", tvmdev, 1, repeat=100)

    # Profile using debug_runtime
    gmodule.run()

    prof_result = timer()
    times = prof_result.results

    result = tvmc.TVMCResult(None, times)

    mean_ts = np.mean(result.times) * 1000
    std_ts = np.std(result.times) * 1000
    max_ts = np.max(result.times) * 1000
    min_ts = np.min(result.times) * 1000

    return {"mean": mean_ts, "std": std_ts, "max": max_ts, "min": min_ts}

def benchmark_tvm_vm_model(
    compiled_model_path: str,
    input_specs: List[InputSpec],
    hostname: Optional[str],
    rpc_key: Optional[str],
    target_device: str,
    profile: bool,
):
    session = common.create_session(hostname, rpc_key)

    dev = device.get_device(target_device)
    tvmdev = common.create_tvmdev(dev.tvmdev, session)

    with tempfile.TemporaryDirectory() as tmp_dir:
        t = tarfile.open(compiled_model_path)
        t.extractall(tmp_dir)
        session.upload(os.path.join(tmp_dir, "lib.tar"))
        lib = session.load_module("lib.tar")
    input_tensors = [
        np.random.uniform(-1, 1, size=ispec.shape).astype(ispec.dtype) for ispec in input_specs
    ]
    input_tensors = [
        tvm.nd.array(input_tensor, tvmdev) for input_tensor in input_tensors
    ]
    from tvm.runtime import vm as rt_vm
    vm = rt_vm.VirtualMachine(lib, tvmdev)

    import time
    elapsed = []
    for i in range(100):
        t1 = time.perf_counter()
        vm.invoke_stateful('main', input_tensors[0])
        t2 = time.perf_counter()
        elapsed.append(t2 - t1)
    elapsed = np.array(elapsed)

    first_ts = elapsed[0] * 1000
    # because first execution is slow, calculate average time from the second execution
    mean_ts = np.mean(elapsed[1:]) * 1000
    std_ts = np.std(elapsed[1:]) * 1000
    max_ts = np.max(elapsed[1:]) * 1000
    min_ts = np.min(elapsed[1:]) * 1000

    return {"first": first_ts, "mean_rest": mean_ts, "std_rest": std_ts, "max_rest": max_ts, "min_rest": min_ts}

def benchmark_for_keras(
    model,  # tf.keras.Model
    compiled_model_path: str,
    hostname: Optional[str],
    rpc_key: Optional[str],
    target_device: str,
    profile: bool,
):
    # TODO: support more inputs
    input_layer = model.get_layer(index=0)
    config = input_layer.get_config()
    input_shape = tuple([1] + list(config["batch_input_shape"][1:]))
    dtype = config["dtype"]

    input_specs = [InputSpec(input_shape, dtype)]

    return benchmark_tvm_model(
        compiled_model_path, input_specs, hostname, rpc_key, target_device, profile
    )


def benchmark_for_onnx_vm(
    model,  # onnx.onnx_ml_pb2.ModelProto
    compiled_model_path: str,
    hostname: Optional[str],
    rpc_key: Optional[str],
    target_device: str,
    profile: bool,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        import onnx
        import onnxruntime
        tmp_model_path = tmp_dir + '/model.onnx'
        onnx.save_model(model, tmp_model_path)
        sess =  onnxruntime.InferenceSession(tmp_model_path)
        inputs = sess.get_inputs()

    input_specs = [InputSpec([1] + input_shape.shape[1:], 'float32') for input_shape in inputs]

    return benchmark_tvm_vm_model(
        compiled_model_path, input_specs, hostname, rpc_key, target_device, profile
    )


def benchmark_for_pytorch(
    model,  # torch.nn.Module
    compiled_model_path: str,
    input_specs: List[InputSpec],
    hostname: Optional[str],
    rpc_key: Optional[str],
    target_device: str,
    profile: bool,
):
    return benchmark_tvm_model(
        compiled_model_path, input_specs, hostname, rpc_key, target_device, profile
    )


def benchmark_for_tf_concrete_function(
    concrete_func,
    compiled_model_path: str,
    hostname: Optional[str],
    rpc_key: Optional[str],
    target_device: str,
    profile: bool,
):

    from tensorflow.python.eager.function import ConcreteFunction

    assert isinstance(concrete_func, ConcreteFunction)

    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    frozen_model, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

    input_specs = []
    for input in frozen_model.inputs:
        input_specs.append(InputSpec(input.shape, str(input.dtype.name)))

    return benchmark_tvm_model(
        compiled_model_path, input_specs, hostname, rpc_key, target_device, profile
    )
