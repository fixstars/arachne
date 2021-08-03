import time
from pathlib import Path

import tensorflow as tf

from arachne.device import get_target
from arachne.pipeline.package.frontend import make_tflite_package
from arachne.pipeline.runner import run_pipeline
from arachne.pipeline.stage.registry import get_stage
from arachne.runtime import runner_init
from arachne.types import IndexedOrderedDict, QType, TensorInfo

# ============================================================================================= #

"""Typicaly, you will modify the following CAPITAL variables"""

# To refer a local file:
# MODEL_URI='file:///workspace/<model-name>.tflite',
MODEL_URI = "https://ion-archives.s3.us-west-2.amazonaws.com/models/tflite/efficientdet-d0.tflite"

# Model-specific input/output tensor information
INPUT_INFO = IndexedOrderedDict(
    [("image_arrays", TensorInfo(shape=[1, 512, 512, 3], dtype="uint8"))]
)
OUTPUT_INFO = IndexedOrderedDict([("detections", TensorInfo(shape=[1, 100, 7], dtype="float32"))])

# An output directory for saving execution results
OUTPUT_DIR = "./out"

# Compile targets
# You can find avaiable varaibles in arachne/device.py
TARGET_DEVICE = "host"
# TARGET_DEVICE = "jetson-nano,cpu"
# TARGET_DEVICE = "jetson-nano,trt,cpu"

# RPC server/tracker hostname
RPC_HOST = None
# RPC_HOST = '<hostname>:<port>'

# RPC key
RPC_KEY = None

# ============================================================================================= #

compile_start = time.time()
print("Compiling... ", end="")

# Create a package
# Arachne manages the inputs/outputs of each compile/model-conversion/quantization as a package
pkg = make_tflite_package(
    model_url=MODEL_URI,
    input_info=INPUT_INFO,
    output_info=OUTPUT_INFO,
    output_dir=Path(OUTPUT_DIR),
    qtype=QType.FP32,
    for_edgetpu=False,
)

# Run a compile pipeline
compile_pipeline = [(get_stage("tvm_compiler"), {})]

target = get_target(TARGET_DEVICE)

default_params = dict()
default_params.update(
    {
        "_compiler_target": target.target,
        "_compiler_target_host": target.target_host,
        "_quantizer_qtype": target.default_qtype,
    }
)
outputs = run_pipeline(compile_pipeline, pkg, default_params, OUTPUT_DIR)

compile_duration = time.time() - compile_start
print("Done! {} sec".format(compile_duration))

# Export an arachne package to a specified tar file
# outputs[-1].export('exported.tar')

# Init runtime module
init_start = time.time()
print("Init runtime... ", end="")

module = runner_init(package=outputs[-1], rpc_tracker=RPC_HOST, rpc_key=RPC_KEY, profile=False)

init_duration = time.time() - init_start
print("Done! {} sec".format(init_duration))


# Benchmarking with dummy inputs
benchmark_start = time.time()
print("Benchmarking... ", end="")

res = module.benchmark(10)
print(res)

benchmark_duration = time.time() - benchmark_start
print("Done! {} sec".format(benchmark_duration))


# Run with a real input
inference_start = time.time()
print("Inferencing... ", end="")


def load_image(image_path, image_size):
    input_data = tf.io.gfile.GFile(image_path, "rb").read()
    image = tf.io.decode_image(input_data, channels=3, dtype=tf.uint8)
    image = tf.image.resize(image, image_size, method="bilinear", antialias=True)
    return tf.expand_dims(tf.cast(image, tf.uint8), 0).numpy()


input = load_image("./testdata/img1.jpg", [512, 512])

INPUT_INFO["image_arrays"] = input
module.set_inputs(INPUT_INFO)
module.run()
prediction = module.get_outputs(OUTPUT_INFO)
# print(prediction)
# postprocess(...)

inference_duration = time.time() - inference_start
print("Done! {} sec".format(inference_duration))
