import os
import tempfile

import arachne.benchmark
import arachne.compile
from arachne.ishape import InputSpec
from torchvision import models


def test_benchmark_for_pytorch():
    resnet18 = models.resnet18(pretrained=True)
    input_shape = (1, 3, 224, 224)
    target_device = "host"
    compile_pipeline = [("tvm_compiler", {})]

    input_spec = [InputSpec(input_shape, "float32")]

    with tempfile.TemporaryDirectory() as tmp_dir:

        compiled_model_path, _ = arachne.compile.compile_for_pytorch(
            resnet18, [InputSpec(input_shape, "float32")], target_device, compile_pipeline, tmp_dir
        )

        print(compiled_model_path)
        arachne.benchmark.benchmark_for_pytorch(
            resnet18,
            compiled_model_path,
            input_spec,
            None,
            None,
            target_device,
            True,
        )


def test_benchmark_for_keras():
    import tensorflow as tf

    mobilenet = tf.keras.applications.mobilenet.MobileNet()
    target_device = "host"
    compile_pipeline = [("tvm_compiler", {})]

    with tempfile.TemporaryDirectory() as tmp_dir:
        compiled_model_path, _ = arachne.compile.compile_for_keras(
            mobilenet, target_device, compile_pipeline, tmp_dir
        )

        arachne.benchmark.benchmark_for_keras(
            mobilenet,
            compiled_model_path,
            None,
            None,
            target_device,
            True,
        )


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

        compiled_model_path, _ = arachne.compile.compile_for_tf_concrete_function(
            concrete_func, target_device, compile_pipeline, tmp_dir
        )

        arachne.benchmark.benchmark_for_tf_concrete_function(
            concrete_func,
            compiled_model_path,
            None,
            None,
            target_device,
            True,
        )
