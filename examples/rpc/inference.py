import tempfile
from pathlib import Path

import tensorflow as tf

from arachne.device import get_target
from arachne.pipeline import Pipeline
from arachne.pipeline.package.frontend import make_tf1_package_from_concrete_func
from arachne.pipeline.package.tf1 import TF1Package
from arachne.pipeline.runner import make_params_for_target, run_pipeline
from arachne.pipeline.stage.registry import get_stage
from arachne.runtime import TVMRuntimeModule, runner_init

RPC_TRACKER = '0.0.0.0:9190'

RPC_KEY = 'jetson-nano'


def inference():
    """ This is an example code for understanding how to use arachne APIs on local machine.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        ### Construct a simple tensor function to be tested
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.float32),
            ]
        )
        def add(x, y):
            return x + y
        cfunc = add.get_concrete_function()

        pkg: TF1Package = make_tf1_package_from_concrete_func(cfunc, Path(tmp_dir))

        ## ==== Unique to RPC usage ====

        ### You have to specify the appropreate device for runtime environment
        target = get_target("jetson-nano")
        assert target is not None
        default_params = make_params_for_target(target)
        tvm_compiler = get_stage("tvm_compiler")
        assert tvm_compiler is not None
        compile_pipeline: Pipeline = [(tvm_compiler, {})]

        outputs = run_pipeline(compile_pipeline, pkg, default_params, tmp_dir)
        compiled_pkg = outputs[-1]

        ### When using RPC, you have to set rpc_tracker and rpc_key information to runner_init()
        mod = runner_init(compiled_pkg, rpc_tracker=RPC_TRACKER, rpc_key=RPC_KEY)
        assert isinstance(mod, TVMRuntimeModule)

        ## ==== End of Unique to RPC usage ====

        mod.set_input(0, 1.0)
        mod.set_input(1, 2.0)
        mod.run()
        print('Index-based usage: output:', mod.get_output(0))

        input = mod.get_input_details()
        input["x"] = tf.constant(1.0)
        input["y"] = tf.constant(2.0)
        mod.set_inputs(input)

        mod.run()

        output_info = mod.get_output_details()
        print('Dict-based usage: output:', mod.get_outputs(output_info))

        ## You can benchmark this model
        r = mod.benchmark(repeat=10)
        print('Benchmark result: ', r)


inference()
