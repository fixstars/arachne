import os
import tempfile

from torchvision import models

import arachne.benchmark
import arachne.compile
from arachne.ishape import InputSpec


def test_benchmark_for_pytorch():
    resnet18 = models.resnet18(pretrained=True)
    input_shape = (1, 3, 224, 224)
    target_device = 'host'
    pipeline = 'tvm'

    input_spec = [InputSpec(input_shape, 'float32')]

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'resent18.tar')

        arachne.compile.compile_for_pytorch(
            resnet18,
            input_spec,
            target_device,
            pipeline,
            output_path
        )

        arachne.benchmark.benchmark_for_pytorch(
            resnet18,
            output_path,
            input_spec,
            None,
            None,
            target_device,
            True,
        )

def test_benchmark_for_keras():
    import tensorflow as tf
    mobilenet = tf.keras.applications.mobilenet.MobileNet()
    target_device = 'host'
    pipeline = 'tvm'

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'keras-mobilenet.tar')

        arachne.compile.compile_for_keras(
            mobilenet,
            target_device,
            pipeline,
            output_path
        )

        arachne.benchmark.benchmark_for_keras(
            mobilenet,
            output_path,
            None,
            None,
            target_device,
            True,
        )
