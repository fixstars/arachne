import os
import tempfile

import arachne.compile
from arachne.ishape import InputSpec
from torchvision import models


def test_compile_for_pytorch():
    resnet18 = models.resnet18(pretrained=True)
    input_shape = (1, 3, 224, 224)
    target_device = "host"
    pipeline = "tvm"

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "resent18.tar")

        arachne.compile.compile_for_pytorch(
            resnet18, [InputSpec(input_shape, "float32")], target_device, pipeline, output_path
        )


def test_compile_for_keras():
    import tensorflow as tf

    mobilenet = tf.keras.applications.mobilenet.MobileNet()
    target_device = "host"
    pipeline = "tvm"

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "keras-mobilenet.tar")

        arachne.compile.compile_for_keras(mobilenet, target_device, pipeline, output_path)


def test_compile_for_tf_concrete_function():
    import tensorflow as tf

    mobilenet = tf.keras.applications.mobilenet.MobileNet()
    target_device = "host"
    pipeline = "tvm"

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32)])
    def wrap(x):
        return mobilenet(x)

    concrete_func = wrap.get_concrete_function()

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "tf-mobilenet-concrete-func.tar")

        arachne.compile.compile_for_tf_concrete_function(
            concrete_func, target_device, pipeline, output_path
        )
