import tempfile

import pytest
import tensorflow as tf
import tensorflow_datasets as tfds

import arachne.compile


@pytest.mark.parametrize("qtype", ["fp32", "fp16", "int8"])
def test_tflite_converter_from_keras(qtype: str):
    """Tests for tflite converter from tf.keras.Model"""
    mobilenet = tf.keras.applications.mobilenet.MobileNet()
    target_device = "host"

    def make_dataset():
        return tfds.load(
            name="coco/2017", split="validation", data_dir="/datasets/TFDS", download=False
        )

    def preprocess(image):
        preprocessed = tf.cast(image, dtype=tf.float32)
        preprocessed = tf.image.resize(image, [224, 224])
        preprocessed = tf.keras.applications.mobilenet.preprocess_input(preprocessed)
        preprocessed = tf.expand_dims(preprocessed, axis=0)
        return preprocessed

    tflite_converter_param = {
        "qtype": qtype,
        "make_dataset": make_dataset,
        "qsample": 100,
        "preprocess": preprocess,
    }

    compile_pipeline = [("tflite_converter", tflite_converter_param), ("tvm_compiler", {})]
    with tempfile.TemporaryDirectory() as tmp_dir:
        arachne.compile.compile_for_keras(mobilenet, target_device, compile_pipeline, tmp_dir)


@pytest.mark.parametrize("qtype", ["fp32", "fp16", "int8"])
def test_tflite_converter_from_tf_concrete_function(qtype: str):
    """Tests for tflite converter from tf concrete function"""
    mobilenet = tf.keras.applications.mobilenet.MobileNet()
    target_device = "host"

    def make_dataset():
        return tfds.load(
            name="coco/2017", split="validation", data_dir="/datasets/TFDS", download=False
        )

    def preprocess(image):
        preprocessed = tf.cast(image, dtype=tf.float32)
        preprocessed = tf.image.resize(image, [224, 224])
        preprocessed = tf.keras.applications.mobilenet.preprocess_input(preprocessed)
        preprocessed = tf.expand_dims(preprocessed, axis=0)
        return preprocessed

    tflite_converter_param = {
        "qtype": qtype,
        "make_dataset": make_dataset,
        "qsample": 100,
        "preprocess": preprocess,
    }

    compile_pipeline = [("tflite_converter", tflite_converter_param), ("tvm_compiler", {})]
    with tempfile.TemporaryDirectory() as tmp_dir:

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 224, 224, 3), dtype=tf.float32)])
        def wrap(x):
            return mobilenet(x)

        concrete_func = wrap.get_concrete_function()
        arachne.compile.compile_for_tf_concrete_function(
            concrete_func, target_device, compile_pipeline, tmp_dir
        )
