import tempfile

import numpy as np

from arachne.device import get_target
from arachne.pipeline.package.frontend import make_tf1_package_from_concrete_func
from arachne.pipeline.runner import run_pipeline
from arachne.pipeline.stage.registry import get_stage
from arachne.runtime import runner_init
from arachne.types.qtype import QType


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

        pkg = make_tf1_package_from_concrete_func(concrete_func, tmp_dir)
        pipeline = [(get_stage("tvm_compiler"), {})]

        target = get_target("host")
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

        pkg = make_tf1_package_from_concrete_func(concrete_func, tmp_dir)
        pipeline = [(get_stage("tflite_converter"), {"qtype": QType.FP32})]

        pkg = run_pipeline(pipeline, pkg, {}, tmp_dir)[-1]

        # NOTE: dtype is required.
        # Otherwise, tvm will cause a SEGV at set_input
        input_data = np.array(1.0, dtype=np.float32)

        mod = runner_init(pkg)

        mod.set_input(0, input_data)
        mod.set_input(1, input_data)

        mod.run()

        t = mod.get_output(0)
        assert t.numpy() == 2.0

        mod.benchmark(10)

        del t
