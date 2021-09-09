import tempfile
from pathlib import Path

import pytest
import tensorflow as tf
import tensorflow_datasets as tfds

from arachne.pipeline.package.frontend import (
    make_keras_package_from_module,
    make_tf1_package_from_concrete_func,
    make_tf2_package_from_module,
)
from arachne.pipeline.runner import run_pipeline
from arachne.pipeline.stage.registry import get_stage
from arachne.runtime.indexed_ordered_dict import TensorInfoDict
from arachne.runtime.tensor_info import TensorInfo


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


@pytest.mark.parametrize("qtype", ["fp32", "fp16", "int8"])
def test_tflite_converter_from_keras(qtype: str):
    """Tests for tflite converter from tf.keras.Model"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mobilenet = tf.keras.applications.mobilenet.MobileNet()

        tflite_converter_param = {
            "qtype": qtype,
            "make_dataset": make_dataset,
            "qsample": 100,
            "preprocess": preprocess,
        }

        pipeline = [(get_stage("tflite_converter"), tflite_converter_param)]
        input_pkg = make_keras_package_from_module(mobilenet, Path(tmp_dir))
        run_pipeline(pipeline, input_pkg, {}, tmp_dir)


@pytest.mark.parametrize("qtype", ["fp32", "fp16", "int8"])
def test_tflite_converter_from_tf1(qtype: str):
    """Tests for tflite converter from tf concrete function"""
    with tempfile.TemporaryDirectory() as tmp_dir:

        mobilenet = tf.keras.applications.mobilenet.MobileNet()

        tflite_converter_param = {
            "qtype": qtype,
            "make_dataset": make_dataset,
            "qsample": 100,
            "preprocess": preprocess,
        }

        @tf.function(input_signature=[tf.TensorSpec(shape=(1, 224, 224, 3), dtype=tf.float32)])
        def wrap(x):
            return mobilenet(x)

        concrete_func = wrap.get_concrete_function()

        pipeline = [(get_stage("tflite_converter"), tflite_converter_param)]
        input_pkg = make_tf1_package_from_concrete_func(concrete_func, Path(tmp_dir))
        run_pipeline(pipeline, input_pkg, {}, tmp_dir)


@pytest.mark.parametrize("qtype", ["fp32", "fp16", "int8"])
def test_tflite_converter_from_tf2(qtype: str):
    """Tests for tflite converter from tf concrete function"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mobilenet = tf.keras.applications.mobilenet.MobileNet()

        tflite_converter_param = {
            "qtype": qtype,
            "make_dataset": make_dataset,
            "qsample": 100,
            "preprocess": preprocess,
        }

        input_info = TensorInfoDict()
        input_info["input"] = TensorInfo(shape=[1, 224, 224, 3])

        output_info = TensorInfoDict()
        output_info["predictions/Softmax:0"] = TensorInfo(shape=[1, 1000])

        pipeline = [(get_stage("tflite_converter"), tflite_converter_param)]
        input_pkg = make_tf2_package_from_module(mobilenet, input_info, output_info, Path(tmp_dir))
        run_pipeline(pipeline, input_pkg, {}, tmp_dir)
