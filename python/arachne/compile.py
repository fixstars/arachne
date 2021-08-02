import tempfile
from pathlib import Path
from typing import List, Tuple

from arachne.device import get_device
from arachne.pipeline.package.frontend import (
    make_keras_package_from_module,
    make_onnx_package_from_module,
    make_tf1_package_from_concrete_func,
    make_torchscript_package_from_script_module,
)
from arachne.pipeline.runner import run_pipeline
from arachne.pipeline.stage.registry import get_stage
from arachne.pipeline.stage.stage import Parameter
from arachne.types.indexed_ordered_dict import TensorInfoDict
from arachne.types.tensor_info import TensorInfo

from .ishape import InputSpec


def compile_for_pytorch(
    model,  # torch.nn.Module
    input_spec: List[InputSpec],
    target_device: str,
    pipeline: List[Tuple[str, Parameter]],
    output_dir: str,
):
    import torch

    assert isinstance(model, torch.nn.Module)

    compile_pipeline = []
    for stage in pipeline:
        compile_pipeline.append((get_stage(stage[0]), stage[1]))

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_info = {
            f"input{n}": TensorInfo(shape=ispec.shape, dtype=ispec.dtype)
            for n, ispec in enumerate(input_spec)
        }

        inps = tuple(torch.zeros(i.shape) for i in input_spec)
        model.eval()
        script_model = torch.jit.trace(model.forward, inps).eval()

        output_info = TensorInfoDict()
        for i, t in enumerate(script_model(*inps)):
            output_info["output" + str(i)] = TensorInfo(
                shape=list(t.shape), dtype=str(t.dtype).split(".")[-1]
            )

        input_pkg = make_torchscript_package_from_script_module(
            script_model, input_info, output_info, Path(tmp_dir)
        )

        # run pipeline
        device = get_device(target_device)

        default_params = dict()
        default_params.update(
            {
                "_compiler_target": device.target,
                "_compiler_target_host": device.target_host,
                "_quantizer_qtype": device.default_dtype,
            }
        )

        outputs = run_pipeline(compile_pipeline, input_pkg, default_params, output_dir)

        # TODO what should be returned by this function?
        last_output = outputs[-1]
        compiled_module_file = str(last_output.dir / last_output.package_file)
        return compiled_module_file, compiled_module_file + ".relay"


# NOTE: model should be a tf.keras.Model
def compile_for_keras(
    model, target_device: str, pipeline: List[Tuple[str, Parameter]], output_dir: str
):
    import tensorflow as tf

    assert isinstance(model, tf.keras.Model)

    compile_pipeline = []
    for stage in pipeline:
        compile_pipeline.append((get_stage(stage[0]), stage[1]))

    with tempfile.TemporaryDirectory() as tmp_dir:
        # construct state input (a package)
        input_pkg = make_keras_package_from_module(model, Path(tmp_dir))

        # run pipeline
        device = get_device(target_device)

        default_params = dict()
        default_params.update(
            {
                "_compiler_target": device.target,
                "_compiler_target_host": device.target_host,
                "_quantizer_qtype": device.default_dtype,
            }
        )

        outputs = run_pipeline(compile_pipeline, input_pkg, default_params, output_dir)

        # TODO what should be returned by this function?
        last_output = outputs[-1]
        compiled_module_file = str(last_output.dir / last_output.package_file)
        return compiled_module_file, compiled_module_file + ".relay"

def compile_for_onnx_vm(
    model, target_device: str, pipeline: List[Tuple[str, Parameter]], output_dir: str
):
    import onnx

    compile_pipeline = []
    for stage in pipeline:
        compile_pipeline.append((get_stage(stage[0]), stage[1]))

    with tempfile.TemporaryDirectory() as tmp_dir:
        # construct state input (a package)
        input_pkg = make_onnx_package_from_module(model, Path(tmp_dir))

        # run pipeline
        device = get_device(target_device)

        default_params = dict()
        default_params.update(
            {
                "_compiler_target": device.target,
                "_compiler_target_host": device.target_host,
                "_quantizer_qtype": device.default_dtype,
            }
        )

        outputs = run_pipeline(compile_pipeline, input_pkg, default_params, output_dir)

        # TODO what should be returned by this function?
        last_output = outputs[-1]
        compiled_module_file = str(last_output.dir / last_output.package_file)
        return compiled_module_file, compiled_module_file + ".relay"


def compile_for_tf_concrete_function(
    concrete_func, target_device: str, pipeline: List[Tuple[str, Parameter]], output_dir: str
):

    from tensorflow.python.eager.function import ConcreteFunction

    assert isinstance(concrete_func, ConcreteFunction)

    compile_pipeline = []
    for stage in pipeline:
        compile_pipeline.append((get_stage(stage[0]), stage[1]))

    with tempfile.TemporaryDirectory() as tmp_dir:
        # construct state input (a package)
        input_pkg = make_tf1_package_from_concrete_func(concrete_func, Path(tmp_dir))

        # run pipeline
        device = get_device(target_device)

        default_params = dict()
        default_params.update(
            {
                "_compiler_target": device.target,
                "_compiler_target_host": device.target_host,
                "_quantizer_qtype": device.default_dtype,
            }
        )

        outputs = run_pipeline(compile_pipeline, input_pkg, default_params, output_dir)

        # TODO what should be returned by this function?
        last_output = outputs[-1]
        compiled_module_file = str(last_output.dir / last_output.package_file)
        return compiled_module_file, compiled_module_file + ".relay"
