import tempfile

from torchvision import models

import arachne.benchmark
import arachne.compile
from arachne.ishape import InputSpec
from arachne.types.indexed_ordered_dict import TensorInfoDict


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

def test_benchmark_for_onnx_vm():
    import onnx
    import torch
    resnet18 = models.resnet18(pretrained=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_input = torch.randn(1, 3, 224, 224)
        onnx_model_path = tmp_dir + '/resnet18.onnx'
        torch.onnx.export(resnet18, dummy_input, onnx_model_path)
        onnx_model = onnx.load_model(onnx_model_path)
        target_device = "host"
        compile_pipeline = [("tvm_vm_compiler", {})]

        compiled_model_path, _ = arachne.compile.compile_for_onnx_vm(
            onnx_model, target_device, compile_pipeline, tmp_dir
        )

        arachne.benchmark.benchmark_for_onnx_vm(
            onnx_model,
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

def test_benchmark_for_tflite_model():
    import tempfile
    from pathlib import Path

    from arachne.device import get_device
    from arachne.pipeline.package.frontend import make_tflite_package
    from arachne.pipeline.runner import run_pipeline
    from arachne.pipeline.stage.registry import get_stage
    from arachne.types import QType, TensorInfo, TensorInfoDict

    with tempfile.TemporaryDirectory() as tmp_dir:

        # Specify input/output tensor information
        input_info = TensorInfoDict([('image_arrays', TensorInfo(shape=[1, 512, 512, 3], dtype='uint8'))])
        output_info = TensorInfoDict([('detections', TensorInfo(shape=[1, 100, 7], dtype='float32'))])

        pkg = make_tflite_package(
            # model_url='file:///workspace/<model-name>.tflite',
            model_url='https://ion-archives.s3.us-west-2.amazonaws.com/models/tflite/efficientdet-d0.tflite',
            input_info=input_info,
            output_info=output_info,
            output_dir=Path(tmp_dir),
            qtype=QType.FP32,
            for_edgetpu=False
        )

        compile_pipeline = [(get_stage('tvm_compiler') , {})]

        # Specify a target device: see arachne/device.py for available devices
        device = get_device("host")

        default_params = dict()
        default_params.update(
            {
                "_compiler_target": device.target,
                "_compiler_target_host": device.target_host,
                "_quantizer_qtype": device.default_dtype,
            }
        )

        outputs = run_pipeline(compile_pipeline, pkg, default_params, tmp_dir)

        compiled_model_path = outputs[-1].dir / outputs[-1].package_file

        ispec = []
        for v in input_info.values():
            ispec.append(InputSpec(shape=v.shape, dtype=v.dtype))

        arachne.benchmark.benchmark_tvm_model(
            compiled_model_path,
            ispec,
            None,
            None,
            'host',
            True,
        )
