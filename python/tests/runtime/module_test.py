import tempfile
from pathlib import Path

import numpy as np

from arachne.device import TVMCTarget, get_target
from arachne.pipeline.package.frontend import make_tf1_package_from_concrete_func
from arachne.pipeline.pipeline import Pipeline
from arachne.pipeline.runner import run_pipeline
from arachne.pipeline.stage.registry import get_stage
from arachne.runtime import runner_init
from arachne.runtime.indexed_ordered_dict import IndexedOrderedDict
from arachne.runtime.module.onnx import ONNXRuntimeModule
from arachne.runtime.module.tflite import TFLiteRuntimeModule
from arachne.runtime.module.tvm import TVMRuntimeModule, TVMVMRuntimeModule
from arachne.runtime.package import ONNXPackage
from arachne.runtime.qtype import QType
from arachne.runtime.tensor_info import TensorInfo


def test_tvm_runtime_module():
    import tensorflow as tf

    with tempfile.TemporaryDirectory() as tmp_dir:

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.float32),
            ]
        )
        def add(x, y):
            return x + y

        concrete_func = add.get_concrete_function()

        pkg = make_tf1_package_from_concrete_func(concrete_func, Path(tmp_dir))
        tvm_compiler = get_stage("tvm_compiler")
        assert tvm_compiler is not None
        pipeline: Pipeline = [(tvm_compiler, {})]

        target = get_target("host")
        assert isinstance(target, TVMCTarget)
        default_params = dict()
        default_params.update(
            {
                "_compiler_target": target.target,
                "_compiler_target_host": target.target_host,
                "_quantizer_qtype": target.default_qtype,
            }
        )

        pkg = run_pipeline(pipeline, pkg, default_params, tmp_dir)[-1]

        mod = runner_init(pkg)
        assert isinstance(mod, TVMRuntimeModule)

        # Check primitive methods work correctly
        assert mod.get_num_inputs() == 2

        mod.set_input(0, 1.0)
        mod.set_input(1, 2.0)

        v1 = mod.get_input(0)
        v2 = mod.get_input(1)

        assert v1.numpy() == 1.0
        assert v2.numpy() == 2.0

        mod.run()

        assert mod.get_num_outputs() == 1

        v3 = mod.get_output(0)
        assert v3.numpy() == 3.0

        # Check get_input_detail(), set_inputs(), get_outputs_details(), and get_outputs()
        input = mod.get_input_details()
        input["x"] = tf.constant(1.0)
        input["y"] = tf.constant(2.0)
        mod.set_inputs(input)

        mod.run()

        output_info = mod.get_output_details()
        outputs = mod.get_outputs(output_info)
        assert outputs["Identity"] == 3.0

        mod.benchmark(10)

        del v1, v2, v3, outputs


def test_tflite_runtime_module():
    import tensorflow as tf

    with tempfile.TemporaryDirectory() as tmp_dir:

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.float32),
            ]
        )
        def add(x, y):
            return x + y

        concrete_func = add.get_concrete_function()

        pkg = make_tf1_package_from_concrete_func(concrete_func, Path(tmp_dir))
        tflite_converter = get_stage("tflite_converter")
        assert tflite_converter is not None
        pipeline = [(tflite_converter, {"qtype": QType.FP32})]

        pkg = run_pipeline(pipeline, pkg, {}, tmp_dir)[-1]

        # NOTE: dtype is required.
        # Otherwise, tvm will cause a SEGV at set_input
        input_data = np.array(1.0, dtype=np.float32)  # type: ignore

        mod = runner_init(pkg)
        assert isinstance(mod, TFLiteRuntimeModule)

        mod.set_input(0, input_data)
        mod.set_input(1, input_data)

        mod.run()

        t = mod.get_output(0)
        assert t.numpy() == 2.0

        mod.benchmark(10)

        del t


def test_tvmvm_runtime_module():
    import tensorflow as tf

    with tempfile.TemporaryDirectory() as tmp_dir:

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.float32),
            ]
        )
        def add(x, y):
            return x + y

        concrete_func = add.get_concrete_function()

        pkg = make_tf1_package_from_concrete_func(concrete_func, Path(tmp_dir))
        tvm_vm_compiler = get_stage("tvm_vm_compiler")
        assert tvm_vm_compiler is not None
        pipeline = [(tvm_vm_compiler, {})]

        target = get_target("host")
        assert isinstance(target, TVMCTarget)
        default_params = dict()
        default_params.update(
            {
                "_compiler_target": target.target,
                "_compiler_target_host": target.target_host,
                "_quantizer_qtype": target.default_qtype,
            }
        )

        pkg = run_pipeline(pipeline, pkg, default_params, tmp_dir)[-1]

        mod = runner_init(pkg, rpc_tracker=None, rpc_key=None, profile=False)
        assert isinstance(mod, TVMVMRuntimeModule)

        # NOTE: dtype is required.
        # Otherwise, tvm will cause a SEGV at set_input
        input_data = np.array(1.0, dtype=np.float32)  # type: ignore
        inputs = [input_data, input_data]

        mod.set_inputs(inputs)

        mod.run()

        outputs = mod.get_output_details()
        t = mod.get_outputs(outputs)
        assert t["Identity"] == 2.0
        del t

        input_data = np.array(2.0, dtype=np.float32)  # type: ignore
        inputs2 = mod.get_input_details()
        inputs2["x"] = input_data
        inputs2["y"] = input_data

        mod.set_inputs(inputs2)

        mod.run()

        outputs = mod.get_output_details()
        t = mod.get_outputs(outputs)
        assert t["Identity"] == 4.0
        del t

        mod.benchmark(10)


def test_onnx_runtime_module():
    from pathlib import Path

    import onnx
    import onnx.checker
    from onnx import TensorProto, helper

    with tempfile.TemporaryDirectory() as tmp_dir:

        node = helper.make_node(
            "Add",
            inputs=["in1", "in2"],
            outputs=["out"],
        )

        graph = helper.make_graph(
            [node],
            "add_test",
            inputs=[
                helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1]),  # type: ignore
                helper.make_tensor_value_info("in2", TensorProto.FLOAT, [1]),  # type: ignore
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])],  # type: ignore
        )

        model = helper.make_model(graph, producer_name="add_test")

        onnx.checker.check_model(model)

        onnx.save(model, tmp_dir + "/test.onnx")

        input_info = IndexedOrderedDict(
            {
                "in1": TensorInfo(shape=[1], dtype="float32"),
                "in2": TensorInfo(shape=[1], dtype="float32"),
            }
        )

        output_info = IndexedOrderedDict({"out": TensorInfo(shape=[1], dtype="float32")})

        pkg: ONNXPackage = ONNXPackage(
            dir=Path(tmp_dir),
            input_info=input_info,
            output_info=output_info,
            model_file=Path("test.onnx"),
        )

        mod = runner_init(pkg, rpc_tracker=None, rpc_key=None, profile=False)
        assert isinstance(mod, ONNXRuntimeModule)

        mod.set_input(0, np.array([1.0], dtype=np.float32))  # type: ignore
        mod.set_input(1, np.array([1.0], dtype=np.float32))  # type: ignore

        mod.run()

        out = mod.get_output(0).numpy()
        assert out == [2.0]
        del out

        mod.benchmark(10)
