import enum
import os
import tempfile
import tarfile
from typing import List, Optional

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

import torch
import pytorch_lightning as pl

import tvm
from tvm.driver import tvmc

from . import device
from .ishape import InputSpec

def benchmark_tvm_model(
    compiled_model_path,
    input_specs: List[InputSpec],
    hostname,
    port,
    target_device
):
    session = tvm.rpc.connect(hostname, port)

    # TODO: support more devices
    dev = session.cpu(0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        t = tarfile.open(compiled_model_path)
        t.extractall(tmp_dir)

        graph = open(os.path.join(tmp_dir, "mod.json")).read()
        params = bytearray(open(os.path.join(tmp_dir, "mod.params"), "rb").read())
        session.upload(os.path.join(tmp_dir, "mod.so"))
        lib = session.load_module("mod.so")

    gmodule = tvm.contrib.graph_executor.create(graph, lib, dev)
    gmodule.load_params(params)

    input_tensors = [np.random.uniform(-1, 1, size=ispec.shape).astype(ispec.dtype) for ispec in input_specs]

    for i, tensor in enumerate(input_tensors):
        gmodule.set_input(i, tensor)
    
    gmodule.run()

    timer = gmodule.module.time_evaluator("run", dev, 1, repeat=100)

    prof_result = timer()
    times = prof_result.results

    result = tvmc.TVMCResult(None, times)

    mean_ts = np.mean(result.times) * 1000
    std_ts = np.std(result.times) * 1000
    max_ts = np.max(result.times) * 1000
    min_ts = np.min(result.times) * 1000

    # stat_table = result.format_times()
    # print(f"\n{stat_table}")
    return {"mean": mean_ts, "std": std_ts, "max": max_ts, "min": min_ts}

def benchmark_for_keras(
    model: tf.keras.Model,
    compiled_model_path: str,
    hostname: str,
    port: int,
    target_device: str,
):
    # TODO: support more inputs
    input_layer = model.get_layer(index=0)
    config = input_layer.get_config()
    input_shape = tuple([1] + list(config['batch_input_shape'][1:]))
    dtype = config['dtype']

    input_specs = [InputSpec(input_shape, dtype)]

    return benchmark_tvm_model(
        compiled_model_path,
        input_specs,
        hostname,
        port,
        target_device
    )

def benchmark_for_pytorch(
    model: torch.nn.Module,
    compiled_model_path: str,
    input_specs: List[InputSpec],
    hostname: str,
    port: str,
    target_device: device.Device,
):
    return benchmark_tvm_model(
        compiled_model_path,
        input_specs,
        hostname,
        port,
        target_device
    )
