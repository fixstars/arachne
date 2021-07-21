import tempfile

from torchvision import models

import arachne.compile
from arachne.ishape import InputSpec


def test_compile_for_pytorch():
    resnet18 = models.resnet18(pretrained=True)
    input_shape = (1, 3, 224, 224)
    target_device = "host"
    compile_pipeline = [("tvm_compiler", {})]

    with tempfile.TemporaryDirectory() as tmp_dir:
        arachne.compile.compile_for_pytorch(
            resnet18, [InputSpec(input_shape, "float32")], target_device, compile_pipeline, tmp_dir
        )


def test_compile_for_keras():
    import tensorflow as tf

    mobilenet = tf.keras.applications.mobilenet.MobileNet()
    target_device = "host"
    compile_pipeline = [("tvm_compiler", {})]
    with tempfile.TemporaryDirectory() as tmp_dir:
        arachne.compile.compile_for_keras(mobilenet, target_device, compile_pipeline, tmp_dir)


def test_compile_for_tf_concrete_function():
    import tensorflow as tf

    mobilenet = tf.keras.applications.mobilenet.MobileNet()
    input_shape = (1, 224, 224, 3)
    target_device = "host"
    compile_pipeline = [("tvm_compiler", {})]
    with tempfile.TemporaryDirectory() as tmp_dir:

        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
        def wrap(x):
            return mobilenet(x)

        concrete_func = wrap.get_concrete_function()
        arachne.compile.compile_for_tf_concrete_function(
            concrete_func, target_device, compile_pipeline, tmp_dir
        )
