import os
import tempfile

import numpy as np
import onnxruntime as ort
import tensorflow as tf
from tvm.contrib.download import download

from arachne.data import Model
from arachne.tools.openvino2tf import OpenVINO2TFConfig
from arachne.tools.openvino2tf import run as run_openvino2tf
from arachne.tools.openvino_mo import OpenVINOModelOptConfig
from arachne.tools.openvino_mo import run as run_openvino_mo
from arachne.utils.model_utils import get_model_spec


def test_openvino2tf():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)

        url = (
            "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/resnet18.onnx"
        )

        onnx_model_path = "resnet18.onnx"
        download(url, onnx_model_path)

        input_model = Model(path=onnx_model_path, spec=get_model_spec(onnx_model_path))
        m = run_openvino_mo(input_model, OpenVINOModelOptConfig())
        m = run_openvino2tf(m, OpenVINO2TFConfig())

        tf_loaded = tf.saved_model.load(m.path)
        resnet18_tf = tf_loaded.signatures["serving_default"]  # type: ignore

        input = np.random.rand(1, 3, 224, 224).astype(np.float32)  # type: ignore

        # onnx runtime
        sess = ort.InferenceSession(onnx_model_path)
        input_name = sess.get_inputs()[0].name
        dout = sess.run(output_names=None, input_feed={input_name: input})[0]

        # tf
        tf_input = tf.convert_to_tensor(np.transpose(input, (0, 2, 3, 1)))
        tf_result = resnet18_tf(tf_input)
        aout = tf_result["tf.identity"].numpy()

        np.testing.assert_allclose(aout, dout, atol=1e-5, rtol=1e-5)
