from pathlib import Path

import tensorflow as tf

from arachne.benchmark import benchmark_tvm_model
from arachne.common import run_module, runner_init
from arachne.device import get_device
from arachne.ishape import InputSpec
from arachne.pipeline.package.frontend import make_tflite_package
from arachne.pipeline.runner import run_pipeline
from arachne.pipeline.stage.registry import get_stage
from arachne.types import QType, TensorInfo, TensorInfoDict

# ============================================================================================= #

"""Typicaly, you will modify the following CAPITAL variables"""

# To refer a local file:
# MODEL_URI='file:///workspace/<model-name>.tflite',
MODEL_URI='https://ion-archives.s3.us-west-2.amazonaws.com/models/tflite/efficientdet-d0.tflite'

# Model-specific input/output tensor information
INPUT_INFO = TensorInfoDict([('image_arrays', TensorInfo(shape=[1, 512, 512, 3], dtype='uint8'))])
OUTPUT_INFO = TensorInfoDict([('detections', TensorInfo(shape=[1, 100, 7], dtype='float32'))])

# An output directory for saving execution results
OUTPUT_DIR='./out'

# Compile targets
# You can find avaiable varaibles in arachne/device.py
TARGET_DEVICE='host'

# RPC server/tracker hostname
RPC_HOST=None

# RPC key
RPC_KEY = None

# ============================================================================================= #

# Create a package
# Arachne manages the inputs/outputs of each compile/model-conversion/quantization as a package
pkg = make_tflite_package(
    model_url=MODEL_URI,
    input_info=INPUT_INFO,
    output_info=OUTPUT_INFO,
    output_dir=Path(OUTPUT_DIR),
    qtype=QType.FP32,
    for_edgetpu=False
)

# Run a compile pipeline
compile_pipeline = [(get_stage('tvm_compiler') , {})]
device = get_device(TARGET_DEVICE)

default_params = dict()
default_params.update(
    {
        "_compiler_target": device.target,
        "_compiler_target_host": device.target_host,
        "_quantizer_qtype": device.default_dtype,
    }
)
outputs = run_pipeline(compile_pipeline, pkg, default_params, OUTPUT_DIR)

# Benchmarking
# the last of outputs is the final result of the compile pipeline
compiled_model_path = outputs[-1].dir / outputs[-1].package_file
ispec = []
for v in INPUT_INFO.values():
    ispec.append(InputSpec(shape=v.shape, dtype=v.dtype))

benchmark_tvm_model(
    compiled_model_path,
    ispec,
    RPC_HOST,
    RPC_KEY,
    TARGET_DEVICE,
    True,
)

# Run with a input tensor
def load_image(image_path, image_size):
  input_data = tf.io.gfile.GFile(image_path, 'rb').read()
  image = tf.io.decode_image(input_data, channels=3, dtype=tf.uint8)
  image = tf.image.resize(
      image, image_size, method='bilinear', antialias=True)
  return tf.expand_dims(tf.cast(image, tf.uint8), 0).numpy()

input = load_image('./testdata/img1.jpg', [512, 512])

mod, tvmdev = runner_init(outputs[-1], device)

INPUT_INFO['image_arrays'] = input
prediction = run_module(mod, tvmdev, INPUT_INFO, OUTPUT_INFO)
# print(prediction)
# model_specific_postprocess(...)
