import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from arachne.data import Model, ModelSpec, TensorSpec
from arachne.tools.tflite_converter import TFLiteConverterConfig, run
from arachne.utils import get_model_spec

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
        np.testing.assert_allclose(aout, dout, atol=1e-5, rtol=1e-5)
    elif ptq_method == "fp16":
        np.testing.assert_allclose(aout, dout, atol=0.1, rtol=0)
    elif ptq_method == "dynamic_range":
        np.testing.assert_allclose(aout, dout, atol=0.2, rtol=0)
    else:
        # skip dummy int8
        pass


@pytest.mark.parametrize("model_format, ptq_method", list(params.values()), ids=list(params.keys()))
def test_tflite_converter(model_format, ptq_method):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        model: tf.keras.Model = tf.keras.applications.mobilenet.MobileNet()
        cfg = TFLiteConverterConfig()
        cfg.ptq.method = ptq_method
        input_shape = [1, 224, 224, 3]
        if model_format == "h5":
            model.save("tmp.h5")
            input = Model("tmp.h5", spec=get_model_spec("tmp.h5"))
            output = run(input=input, cfg=cfg)

        elif model_format == "saved_model":
            model.save("saved_model")
            input = Model("saved_model", spec=get_model_spec("saved_model"))
            output = run(input=input, cfg=cfg)
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

            input = Model("frozen_graph.pb", spec=spec)
            output = run(input=input, cfg=cfg)
        else:
            assert False

        check_tflite_output(model, input_shape, ptq_method, output.path)
