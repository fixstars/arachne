import tempfile

import tensorflow as tf

from arachne.device import get_target
from arachne.pipeline import Pipeline
from arachne.pipeline.package.frontend import make_tf1_package_from_concrete_func
from arachne.pipeline.package.tf1 import TF1Package
from arachne.pipeline.runner import make_params_for_target, run_pipeline
from arachne.pipeline.stage.registry import get_stage
from arachne.runtime import TVMRuntimeModule, runner_init
from arachne.target import TVMCTarget


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

        ## arachne deal with input and output as Packages
        ### We provide some usefule frontend functions for craeting packages from variable dnn models.
        ### Plz refer python/arachne/pipeline/frontend.py
        ### The definition of Packages are python/arachne/pipeline/package/*.py and python/arachne/runtime/package/*.py
        pkg: TF1Package = make_tf1_package_from_concrete_func(cfunc, tmp_dir)

        ## To compile the input model (as a package) for specific devices, you have to prepare the information of target, pipeline, and default parameters.
        target: TVMCTarget = get_target("host")  # create target from the device information (available devices are listed on python/arachne/device.py).
        default_params = make_params_for_target(target)  # create defualt parameters from target

        ## define compile pipeline
        ### arachne define a compile pipline as a list of (stage, params).
        ### avaiable stages are python/arachne/pipeline/stage/*
        compile_pipeline: Pipeline = [(get_stage("tvm_compiler"), {})]

        outputs = run_pipeline(compile_pipeline, pkg, default_params, tmp_dir)
        compiled_pkg = outputs[-1]  # the final result is placed at the last element of outputs

        ## To run inference of the compile model, you first init a runtime module for this model.
        ### avaiable runtime modules are listed in python/arachne/runtime/module/*
        mod: TVMRuntimeModule = runner_init(compiled_pkg)

        # Index-based usage
        mod.set_input(0, 1.0)
        mod.set_input(1, 2.0)
        mod.run()
        print('Index-based usage: output:', mod.get_output(0))

        # Dict-based usage
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
