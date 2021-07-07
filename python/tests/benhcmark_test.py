import os
import tempfile

from torchvision import models

import arachne.benchmark
import arachne.compile
from arachne.ishape import InputSpec


def test_benchmark_for_pytorch():
    resnet18 = models.resnet18(pretrained=True)
    input_shape = (1, 3, 224, 224)
    target_device = 'jetson-xavier-nx'
    pipeline = 'tvm'

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, 'resent18.tar')

        arachne.compile.compile_for_pytorch(
            resnet18,
            [InputSpec(input_shape, 'float32')],
            target_device,
            pipeline,
            output_path
        )

        arachne.benchmark.benchmark_for_pytorch(
            resnet18,
            output_path,
            [InputSpec(input_shape, 'float32')],
            'genesis-jetson-xavier-nx-00.fixstars.com',
            9090,
            'host'
        )
