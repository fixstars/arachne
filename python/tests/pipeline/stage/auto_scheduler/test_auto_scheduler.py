import tempfile
from pathlib import Path

import pytest
import tensorflow as tf
import tensorflow_datasets as tfds
from arachne.device import get_target

from arachne.pipeline.package.frontend import (
    make_keras_package_from_module,
)
from arachne.pipeline.runner import make_params_for_target, run_pipeline
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


def test_auto_scheduler_stage():
    """Test for the auto_scheduler stage"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mobilenet = tf.keras.applications.mobilenet.MobileNet()

        tflite_converter_param = {
            "qtype": "fp32",
            "make_dataset": make_dataset,
            "qsample": 100,
            "preprocess": preprocess,
        }

        target = get_target("host")
        tvm_params = make_params_for_target(target)
        tvm_params["num_measure_trials"] = 30

        pipeline = [
            (get_stage("tflite_converter"), tflite_converter_param),
            (get_stage("auto_scheduler"), tvm_params),
            (get_stage("tvm_compiler"), tvm_params)
        ]
        input_pkg = make_keras_package_from_module(mobilenet, Path(tmp_dir))
        run_pipeline(pipeline, input_pkg, {}, tmp_dir)
