import os
import subprocess
import sys
import tarfile
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from arachne.data import Model, ModelFormat, ModelSpec, TensorSpec
from arachne.tools.tflite_converter import TFLiteConverter, TFLiteConverterConfig
from arachne.utils.model_utils import init_from_dir, init_from_file
from arachne.utils.tf_utils import make_tf_gpu_usage_growth

params = {
    "keras": ("h5", "none"),
    "keras-dynamic-range": ("h5", "dynamic_range"),
    "keras-fp16": ("h5", "fp16"),
    "keras-int8": ("h5", "int8"),
    "saved_model": ("saved_model", "none"),
    "saved_model-dynamic-range": ("saved_model", "dynamic_range"),
    "saved_model-fp16": ("saved_model", "fp16"),
    "saved_model-int8": ("saved_model", "int8"),
    "pb": ("pb", "none"),
    "pb-dynamic-range": ("pb", "dynamic_range"),
    "pb-fp16": ("pb", "fp16"),
    "pb-int8": ("pb", "int8"),
}


def create_dummy_representative_dataset():
    datasets = []
    shape = [1, 224, 224, 3]
    dtype = "float32"
    for _ in range(100):
        datasets.append(np.random.rand(*shape).astype(np.dtype(dtype)))  # type: ignore

    np.save("dummy.npy", datasets)


def check_tflite_output(tf_model, input_shape, ptq_method, tflite_model_path):
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)  # type: ignore
    dout = tf_model(input_data).numpy()  # type: ignore

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    aout = interpreter.get_tensor(output_details[0]["index"])

    if ptq_method == "none":
        np.testing.assert_allclose(aout, dout, atol=1e-5, rtol=1e-5)  # type: ignore
    elif ptq_method == "fp16":
        np.testing.assert_allclose(aout, dout, atol=0.1, rtol=0)  # type: ignore
    elif ptq_method == "dynamic_range":
        np.testing.assert_allclose(aout, dout, atol=0.2, rtol=0)  # type: ignore
    else:
        # skip dummy int8
        pass


@pytest.mark.parametrize("model_format, ptq_method", list(params.values()), ids=list(params.keys()))
def test_tflite_converter(model_format, ptq_method):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        model = tf.keras.applications.mobilenet.MobileNet()
        cfg = TFLiteConverterConfig()
        cfg.ptq.method = ptq_method
        if ptq_method == "int8":
            create_dummy_representative_dataset()
            cfg.ptq.representative_dataset = "dummy.npy"
        input_shape = [1, 224, 224, 3]
        if model_format == "h5":
            model.save("tmp.h5")
            input = init_from_file("tmp.h5")
            output = TFLiteConverter.run(input=input, cfg=cfg)

        elif model_format == "saved_model":
            model.save("saved_model")
            input = init_from_dir("saved_model")
            output = TFLiteConverter.run(input=input, cfg=cfg)
        elif model_format == "pb":
            wrapper = tf.function(lambda x: model(x))
            wrapper = wrapper.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))  # type: ignore

            # Get frozen ConcreteFunction
            from tensorflow.python.framework.convert_to_constants import (
                convert_variables_to_constants_v2,
            )

            frozen_func = convert_variables_to_constants_v2(wrapper)
            frozen_func.graph.as_graph_def()

            tf.io.write_graph(
                graph_or_graph_def=frozen_func.graph,
                logdir=".",
                name="frozen_graph.pb",
                as_text=False,
            )

            inputs = []
            outputs = []
            for inp in frozen_func.inputs:
                shape = [1 if x is None else x for x in inp.shape]
                inputs.append(
                    TensorSpec(name=inp.name.split(":")[0], shape=shape, dtype=inp.dtype.name)
                )
            for out in frozen_func.outputs:
                shape = [1 if x is None else x for x in out.shape]
                outputs.append(
                    TensorSpec(name=out.name.split(":")[0], shape=shape, dtype=out.dtype.name)
                )
            spec = ModelSpec(inputs=inputs, outputs=outputs)

            input = Model("frozen_graph.pb", format=ModelFormat.TF_PB, spec=spec)
            output = TFLiteConverter.run(input=input, cfg=cfg)
        else:
            assert False

        check_tflite_output(model, input_shape, ptq_method, output.path)


def test_cli():
    # Due to the test time, we only test one case

    make_tf_gpu_usage_growth()

    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        model = tf.keras.applications.mobilenet.MobileNet()
        model.save("saved_model")

        ret = subprocess.run(
            [
                sys.executable,
                "-m",
                "arachne.driver.cli",
                "+tools=tflite_converter",
                "model_dir=saved_model",
                "output_path=output.tar",
            ]
        )

        assert ret.returncode == 0

        model_path = None
        with tarfile.open("output.tar", "r:gz") as tar:
            for m in tar.getmembers():
                if m.name.endswith(".tflite"):
                    model_path = m.name
            tar.extractall(".")

        assert model_path is not None
        check_tflite_output(model, [1, 224, 224, 3], "none", model_path)
