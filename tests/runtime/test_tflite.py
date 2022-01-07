# import os
import tempfile

import numpy as np
import tensorflow as tf
from tvm.contrib.download import download

import arachne.runtime


def test_tflite_runtime():
    with tempfile.TemporaryDirectory() as tmp_dir:
        url = "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/mobilenet.tflite"

        model_path = tmp_dir + "/mobilenet.tflite"
        download(url, model_path)

        # TFLite Interpreter
        interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=4)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]["shape"]
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)  # type: ignore
        interpreter.set_tensor(input_details[0]["index"], input_data)

        interpreter.invoke()

        tout = interpreter.get_tensor(output_details[0]["index"])

        # Arachne Runtime
        tflite_interpreter_opts = {"num_threads": 4}
        runtime_module = arachne.runtime.init(model_file=model_path, **tflite_interpreter_opts)
        runtime_module.set_input(0, input_data)
        runtime_module.run()
        aout = runtime_module.get_output(0)

        np.testing.assert_equal(actual=aout, desired=tout)

        runtime_module.benchmark()
