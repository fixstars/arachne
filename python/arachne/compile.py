import logging
import os
import tempfile
import shutil
from typing import Dict, List, Tuple

from tvm.driver import tvmc

from . import device
from .ishape import InputSpec


def compile_by_tvm(
    model_path: str,
    frontend: str,
    input_shape_dict: Dict[str, Tuple],
    output_tensors: List[str],
    target_device: str,
    output_path: str
):
    dev = device.get_device(target_device)

    if frontend is 'pb':
        # NOTE: if there are multiple outputs in a frozen graph, we have to specify their names to tvmc
        tvm_model = tvmc.frontends.load_model(
            model_path, frontend, input_shape_dict, outputs=output_tensors)
    else:
        tvm_model = tvmc.frontends.load_model(
            model_path, frontend, input_shape_dict)

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


def compile_for_pytorch(
    model, # torch.nn.Module
    input_spec: List[InputSpec],
    target_device: str,
    pipeline: str,
    output_path: str,
):
    import torch
    # TODO: support more compile pipelines
    assert(pipeline == 'tvm')

    input_shape_dict = {f'input{n}': ispec.shape for n,
                        ispec in enumerate(input_spec)}

    inps = tuple(torch.zeros(shape) for shape in input_shape_dict.values())

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_name = os.path.basename(output_path)
        pt_script_path = os.path.join(tmp_dir, output_name + ".pth")

        model.eval()
        script_model = torch.jit.trace(model.forward, inps).eval()
        script_model.save(pt_script_path)

        return compile_by_tvm(pt_script_path, 'pytorch', input_shape_dict, None, target_device, output_path)


def compile_for_keras(
    model, # tf.keras.Model
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

        return compile_by_tvm(h5_path, model_format, input_shape_dict, None, target_device, output_path)

def compile_for_tf_concrete_function(
    concrete_func,
    target_device: str,
    pipeline: str,
    output_path: str):

    from tensorflow.python.eager.function import ConcreteFunction
    assert(isinstance(concrete_func, ConcreteFunction))

    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    frozen_model, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tf.io.write_graph(
            graph_or_graph_def=graph_def,
            logdir=tmp_dir,
            name="frozen_graph.pb",
            as_text=False,
        )

        input_shape_dict = {}
        for input in frozen_model.inputs:
            input_shape_dict[input.name] = input.shape.as_list()

        output_tensors = []
        for output in frozen_model.outputs:
            output_tensors.append(output.name)

        frozen_pb_file = tmp_dir + "/frozen_graph.pb"

        return compile_by_tvm(frozen_pb_file, "pb", input_shape_dict, output_tensors, target_device, output_path)
