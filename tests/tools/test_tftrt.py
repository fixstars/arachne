import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from arachne.data import Model
from arachne.tools.tftrt import TFTRTConfig, run
from arachne.utils.model_utils import get_model_spec


def create_dummy_representative_dataset():
    datasets = []
    shape = [1, 224, 224, 3]
    dtype = "float32"
    for _ in range(100):
        datasets.append(np.random.rand(*shape).astype(np.dtype(dtype)))  # type: ignore

    np.save("dummy.npy", datasets)


def check_tftrt_output(tf_model, input_shape, precision, tftrt_model_path):
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)  # type: ignore
    dout = tf_model(input_data).numpy()  # type: ignore

    loaded = tf.saved_model.load(tftrt_model_path)

    infer = loaded.signatures["serving_default"]  # type: ignore
    aout = infer(tf.constant(input_data))["predictions"].numpy()

    if precision == "FP32":
        np.testing.assert_allclose(aout, dout, atol=1e-5, rtol=1e-5)
    elif precision == "FP16":
        np.testing.assert_allclose(aout, dout, atol=0.1, rtol=0)
    elif precision == "INT8":
        # skip dummy int8
        pass


@pytest.mark.parametrize("precision", ["FP32", "FP16", "INT8"])
def test_tftrt(precision):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        model = tf.keras.applications.mobilenet.MobileNet()
        model.save("saved_model")

        input_model = Model(path="saved_model", spec=get_model_spec("saved_model"))
        cfg = TFTRTConfig()
        cfg.precision_mode = precision
        if precision == "INT8":
            create_dummy_representative_dataset()
            cfg.representative_dataset = "dummy.npy"

        output = run(input_model, cfg)
        check_tftrt_output(model, [1, 224, 224, 3], precision, output.path)
