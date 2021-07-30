from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

from arachne.benchmark import benchmark_tvm_model
from arachne.common import run_module, runner_init
from arachne.device import get_device, get_target, parse_device_name
from arachne.ishape import InputSpec
from arachne.pipeline.package.frontend import make_tf1_package
from arachne.pipeline.runner import make_pipeline_candidate, run_pipeline
from arachne.pipeline.stage.registry import get_stage
from arachne.types import QType, TensorInfo, TensorInfoDict

# ============================================================================================= #

"""Typicaly, you will modify the following CAPITAL variables"""

# To refer a local file:
# MODEL_URI='file:///workspace/<model-name>.tflite',
MODEL_URI = "https://ion-archives.s3-us-west-2.amazonaws.com/models/tf1_object_detection/ssd_mobilenet_v1_coco_2018_01_28/tflite.pb"

# Model-specific input/output tensor information
INPUT_INFO = TensorInfoDict([("normalized_input_image_tensor", TensorInfo(shape=(1, 300, 300, 3)))])
OUTPUT_INFO = TensorInfoDict(
    [
        (
            "TFLite_Detection_PostProcess",
            TensorInfo([1, 100, 4]),
        ),
        (
            "TFLite_Detection_PostProcess:1",
            TensorInfo([1, 100]),
        ),
        (
            "TFLite_Detection_PostProcess:2",
            TensorInfo([1, 100]),
        ),
        ("TFLite_Detection_PostProcess:3", TensorInfo([1], "int")),
    ]
)

# An output directory for saving execution results
OUTPUT_DIR = "./out"

# Compile targets
# You can find avaiable varaibles in arachne/device.py
TARGET_DEVICE = "host"

# RPC server/tracker hostname
RPC_HOST = None

# RPC key
RPC_KEY = None

# ============================================================================================= #

# Create a package
# Arachne manages the inputs/outputs of each compile/model-conversion/quantization as a package
pkg = make_tf1_package(
    model_url=MODEL_URI,
    input_info=INPUT_INFO,
    output_info=OUTPUT_INFO,
    output_dir=Path(OUTPUT_DIR),
)

target = get_target(TARGET_DEVICE)


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


default_params = dict()
default_params.update(
    {
        "make_dataset": make_dataset,
        "preprocess": preprocess,
    }
)

pipelines = make_pipeline_candidate(pkg, [target], default_params)

for pipeline in pipelines:
    print(pipeline)

    # TODO: add a sample code for running each pipeline
