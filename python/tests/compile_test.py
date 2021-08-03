import tempfile

import arachne.compile
from arachne.types.tensor_info import TensorInfo


def test_compile_for_pytorch():
    from torchvision import models

    resnet18 = models.resnet18(pretrained=True)
    input_shape = (1, 3, 224, 224)
    target_device = "host"
    compile_pipeline = [("tvm_compiler", {})]

    with tempfile.TemporaryDirectory() as tmp_dir:
        arachne.compile.compile_for_pytorch(
            resnet18, [TensorInfo(input_shape, "float32")], target_device, compile_pipeline, tmp_dir
        )

def test_compile_for_onnx_vm():
    import onnx
    import torch
    from torchvision import models

    resnet18 = models.resnet18(pretrained=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_input = torch.randn(1, 3, 224, 224)
        onnx_model_path = tmp_dir + '/resnet18.onnx'
        torch.onnx.export(resnet18, dummy_input, onnx_model_path)
        onnx_model = onnx.load_model(onnx_model_path)
        target_device = "host"
        compile_pipeline = [("tvm_vm_compiler", {})]
        arachne.compile.compile_for_onnx(onnx_model, target_device, compile_pipeline, tmp_dir)

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
