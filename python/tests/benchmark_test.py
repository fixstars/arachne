import tempfile

import arachne.compile
from arachne.runtime import runner_init
from arachne.types import TensorInfo


def test_benchmark_for_pytorch():
    from torchvision import models

    resnet18 = models.resnet18(pretrained=True)
    input_shape = (1, 3, 224, 224)
    target_device = "host"
    compile_pipeline = [("tvm_compiler", {})]

    input_spec = [TensorInfo(input_shape, "float32")]

    with tempfile.TemporaryDirectory() as tmp_dir:

        compiled_packages = arachne.compile.compile_for_pytorch(
            resnet18, input_spec, target_device, compile_pipeline, tmp_dir
        )

        mod = runner_init(
            package=compiled_packages[-1], rpc_tracker=None, rpc_key=None, profile=True
        )
        mod.benchmark(repeat=10)


def test_benchmark_for_keras():
    import tensorflow as tf

    mobilenet = tf.keras.applications.mobilenet.MobileNet()
    target_device = "host"
    compile_pipeline = [("tvm_compiler", {})]

    with tempfile.TemporaryDirectory() as tmp_dir:
        compiled_packages = arachne.compile.compile_for_keras(
            mobilenet, target_device, compile_pipeline, tmp_dir
        )

        mod = runner_init(
            package=compiled_packages[-1], rpc_tracker=None, rpc_key=None, profile=True
        )
        mod.benchmark(repeat=10)


def test_benchmark_for_tf_concrete_function():
    import tensorflow as tf

    mobilenet = tf.keras.applications.mobilenet.MobileNet()
    target_device = "host"
    compile_pipeline = [("tvm_compiler", {})]

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32)])
    def wrap(x):
        return mobilenet(x)

    concrete_func = wrap.get_concrete_function()

    with tempfile.TemporaryDirectory() as tmp_dir:

        compiled_packages = arachne.compile.compile_for_tf_concrete_function(
            concrete_func, target_device, compile_pipeline, tmp_dir
        )

        mod = runner_init(
            package=compiled_packages[-1], rpc_tracker=None, rpc_key=None, profile=True
        )
        mod.benchmark(repeat=10)
