import logging
import os
import tempfile
import shutil
from typing import Dict, List, Tuple

from tvm.driver import tvmc
import tensorflow as tf
import torch

from . import device
from .ishape import InputSpec

def compile_for_pytorch(
    model: torch.nn.Module,
    input_spec: List[InputSpec],
    target_device: str,
    pipeline: str,
    output_path: str,
):
    # TODO: support more compile pipelines
    assert(pipeline == 'tvm')

    input_shape_dict = { f'input{n}': ispec.shape for n, ispec in enumerate(input_spec) }

    inps = tuple(torch.zeros(shape) for shape in input_shape_dict.values())

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_name = os.path.basename(output_path)
        pt_script_path = os.path.join(tmp_dir, output_name + ".pth")

        model.eval()
        script_model = torch.jit.trace(model.forward, inps).eval()
        script_model.save(pt_script_path)

        return compile_by_tvm(pt_script_path, 'pytorch', input_shape_dict, target_device, output_path)

def compile_for_keras(
    model: tf.keras.Model,
    target_device: str,
    pipeline: str,
    output_path: str
):
    # TODO: support more compile pipelines
    assert(pipeline == "tvm")

    model_format = "keras"
    # NOTE assume 1 input layer
    input_layer = model.get_layer(index=0)
    config = input_layer.get_config()
    input_shape = tuple([1] + list(config["batch_input_shape"][1:]))
    input_shape_dict = {config["name"]: input_shape}
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = os.path.join(tmp_dir, model.name + ".h5")
        model.save(h5_path)

        return compile_by_tvm(h5_path, model_format, input_shape_dict, target_device, output_path)

def compile_by_tvm(
    model_path: str,
    frontend: str,
    input_shape_dict: Dict[str, Tuple],
    target_device: str,
    output_path: str
):
    dev = device.get_device(target_device)

    tvm_model = tvmc.frontends.load_model(model_path, frontend, input_shape_dict)

    tvmc.compiler.compile_model(
        tvm_model,
        dev.target,
        package_path=output_path,
        cross=dev.cross_compiler,
        dump_code='relay',
        target_host=dev.target_host,
        desired_layout=None
    )

    return (output_path, output_path + '.relay')
