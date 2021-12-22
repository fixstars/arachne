import os
import tempfile

import pytest
import tensorflow as tf

from arachne.data import Model
from arachne.tools.tftrt import TFTRTConfig, run
from arachne.utils import get_model_spec


@pytest.mark.parametrize("precision", ["FP32", "FP16", "INT8"])
def test_tftrt(precision):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        model = tf.keras.applications.mobilenet.MobileNet()
        model.save("saved_model")

        input_model = Model(file="saved_model", spec=get_model_spec("saved_model"))
        cfg = TFTRTConfig()
        cfg.precision_mode = precision
        run(input_model, cfg)
