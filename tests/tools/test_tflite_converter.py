import os
import tempfile

import pytest
import tensorflow as tf

from arachne.data import Model, ModelSpec, TensorSpec
from arachne.tools.tflite_converter import TFLiteConverterConfig, run
from arachne.utils import get_model_spec

params = {
    "keras": ('h5', 'none'),
    "keras-dynamic-range": ('h5', 'dynamic_range'),
    "keras-fp16": ('h5', 'fp16'),
    "keras-int8": ('h5', 'int8'),
    "saved_model": ("saved_model", "none"),
    "saved_model-dynamic-range": ('saved_model', 'dynamic_range'),
    "saved_model-fp16": ('saved_model', 'fp16'),
    "saved_model-int8": ('saved_model', 'int8'),
    "pb": ('pb', 'none'),
    "pb-dynamic-range": ('pb', 'dynamic_range'),
    "pb-fp16": ('pb', 'fp16'),
    "pb-int8": ('pb', 'int8'),
}


@pytest.mark.parametrize("model_format, ptq_method", list(params.values()), ids=list(params.keys()))
def test_tflite_converter(model_format, ptq_method):
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        model: tf.keras.Model = tf.keras.applications.mobilenet.MobileNet()
        cfg = TFLiteConverterConfig()
        cfg.ptq.method = ptq_method
        if model_format == "h5":
            model.save("tmp.h5")
            input = Model("tmp.h5", spec=get_model_spec("tmp.h5"))
            run(input=input, cfg=cfg)
        elif model_format == "saved_model":
            model.save("saved_model")
            input = Model("saved_model", spec=get_model_spec("saved_model"))
            run(input=input, cfg=cfg)
        elif model_format == "pb":
            wrapper = tf.function(lambda x: model(x))
            wrapper = wrapper.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))  # type: ignore

            # Get frozen ConcreteFunction
            from tensorflow.python.framework.convert_to_constants import (
                convert_variables_to_constants_v2,
            )

            frozen_func = convert_variables_to_constants_v2(wrapper)
            frozen_func.graph.as_graph_def()

            tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=".", name="frozen_graph.pb", as_text=False)

            inputs = []
            outputs = []
            for inp in frozen_func.inputs:
                shape = [1 if x is None else x for x in inp.shape]
                inputs.append(TensorSpec(name=inp.name.split(':')[0], shape=shape, dtype=inp.dtype.name))
            for out in frozen_func.outputs:
                shape = [1 if x is None else x for x in out.shape]
                outputs.append(TensorSpec(name=out.name.split(':')[0], shape=shape, dtype=out.dtype.name))
            spec = ModelSpec(inputs=inputs, outputs=outputs)

            input = Model("frozen_graph.pb", spec=spec)
            run(input=input, cfg=cfg)
