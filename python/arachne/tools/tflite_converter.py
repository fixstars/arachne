import itertools
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import tensorflow as tf
from hydra.utils import to_absolute_path

from arachne.tools.factory import (
    ToolBase,
    ToolConfigBase,
    ToolConfigFactory,
    ToolFactory,
)
from arachne.utils.model_utils import get_model_spec

from ..data import Model, TensorSpec

_FACTORY_KEY = "tflite_converter"


@dataclass
class TFLiteConverterPTQConfg:
    """This class controls the behavior of the Post-training quantization (PTQ).
    For more details, please refer to the TFLite documentation.
    https://www.tensorflow.org/lite/performance/post_training_quantization

    Attributes:
        method  (str): Specifies the PTQ method. Possible variables are "none", "dynamic_range", "fp16", or "int8".

        representative_dataset: A path to calibration dataset (*.npy). Default value is None.
    """
    method: str = "none"
    representative_dataset: Optional[str] = None


@ToolConfigFactory.register(_FACTORY_KEY)
@dataclass
class TFLiteConverterConfig(ToolConfigBase):
    """This class configures the behavior of the TFLite converter.
    For more details, please refer to the TFLite documentation.
    https://www.tensorflow.org/lite/convert

    Attributes:
        enable_tf_ops (bool): Whether to allow Tensorflow Operations in the output model. Default value is False.

        allow_custom_ops (bool): Whether to allow custom operators. Default value is True.

        ptq (TFLiteConverterPTQConfg): Controls the behavior of the post-training quantization.
    """
    enable_tf_ops: bool = False
    allow_custom_ops: bool = True
    ptq: TFLiteConverterPTQConfg = TFLiteConverterPTQConfg()


def _init_tflite_converter(input: Model):
    input_path = input.path
    converter = None
    if input_path.endswith(".h5"):
        # keras
        model = tf.keras.models.load_model(input_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    elif input_path.endswith(".pb"):
        assert (
            input.spec is not None
        ), "To convert *.pb file, you should specify model spec by model_spec=</path/to/spec.yaml>"
        inputs = [inp.name for inp in input.spec.inputs]
        outputs = [out.name for out in input.spec.outputs]
        input_shapes = {}
        for inp in input.spec.inputs:
            input_shapes[inp.name] = inp.shape

        converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
            graph_def_file=input_path,
            input_arrays=inputs,
            output_arrays=outputs,
            input_shapes=input_shapes,
        )
    elif input_path.endswith("saved_model"):
        converter = tf.lite.TFLiteConverter.from_saved_model(input_path)
    else:
        assert False

    return converter


@ToolFactory.register(_FACTORY_KEY)
class TFLiteConverter(ToolBase):
    """This is a runner class for executing the TFLite Converter.
    """
    @staticmethod
    def run(input: Model, cfg: TFLiteConverterConfig) -> Model:
        """
        The run method is a static method that executes the TFLite Converter for an input model.

        Args:
            input (Model): An input model.
            cfg (TFLiteConverterConfig): A config object.
        Returns:
            Model: A TFLite Model.
        """
        idx = itertools.count().__next__()
        converter = _init_tflite_converter(input)
        assert converter is not None

        converter.allow_custom_ops = cfg.allow_custom_ops
        if cfg.enable_tf_ops:
            converter.target_spec.supported_ops.add(tf.lite.OpsSet.SELECT_TF_OPS)

        if cfg.ptq.method != "none":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # type: ignore
            if cfg.ptq.method == "dynamic_range":
                pass
            elif cfg.ptq.method == "fp16":
                converter.target_spec.supported_types = [tf.float16]
            elif cfg.ptq.method == "int8":
                # TODO support multiple inputs
                datasets: List[np.ndarray]
                if cfg.ptq.representative_dataset is not None:
                    datasets = np.load(to_absolute_path(cfg.ptq.representative_dataset))  # type: ignore
                else:
                    datasets = []
                    assert input.spec is not None
                    inputs: List[TensorSpec] = input.spec.inputs
                    shape = [1 if d == -1 else d for d in inputs[0].shape]
                    dtype = inputs[0].dtype
                    for _ in range(100):
                        datasets.append(np.random.rand(*shape).astype(np.dtype(dtype)))  # type: ignore

                def representative_dataset():
                    for data in datasets:
                        yield [data]

                converter.representative_dataset = representative_dataset  # type: ignore

        tflite_model = converter.convert()

        filename = f"model_{idx}.tflite"
        output_path = os.getcwd() + "/" + filename
        with open(output_path, "wb") as w:
            w.write(tflite_model)

        return Model(path=output_path, spec=get_model_spec(output_path))
