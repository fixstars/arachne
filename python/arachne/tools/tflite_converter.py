import itertools
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

import hydra
import numpy as np
import tensorflow as tf
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING, DictConfig, OmegaConf

from arachne.utils import (
    get_model_spec,
    get_tool_config_objects,
    get_tool_run_objects,
    load_model_spec,
    save_model,
)

from ..data import Model, TensorSpec


@dataclass
class TFLiteConverterPTQConfg:
    # We cannot use Literal['none', 'dynamic_range', 'fp16', 'int8'] due to the limitation of omegaconfig
    method: str = "none"
    representative_dataset: Optional[str] = None


@dataclass
class TFLiteConverterConfig:
    enable_tf_ops: bool = False
    allow_custom_ops: bool = True
    ptq: TFLiteConverterPTQConfg = TFLiteConverterPTQConfg()


def register_tflite_converter_config() -> None:
    cs = ConfigStore.instance()
    group_name = "tools"
    cs.store(
        group=group_name,
        name="tflite_converter",
        package="tools.tflite_converter",
        node=TFLiteConverterConfig,
    )


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


def run(input: Model, cfg: TFLiteConverterConfig) -> Model:
    idx = itertools.count().__next__()
    converter = _init_tflite_converter(input)
    assert converter is not None

    converter.allow_custom_ops = cfg.allow_custom_ops
    if cfg.enable_tf_ops:
        converter.target_spec.supported_ops.add(tf.lite.OpsSet.SELECT_TF_OPS)

    if cfg.ptq.method != "none":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
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


@hydra.main(config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_model_path = to_absolute_path(cfg.input)
    output_path = to_absolute_path(cfg.output)

    input_model = Model(path=input_model_path, spec=get_model_spec(input_model_path))

    # overwrite model spec if input_spec is specified
    if cfg.input_spec:
        input_model.spec = load_model_spec(to_absolute_path(cfg.input_spec))

    assert input_model.spec is not None
    output_model = run(input=input_model, cfg=cfg.tools.tflite_converter)
    save_model(model=output_model, output_path=output_path)


if __name__ == "__main__":
    register_tflite_converter_config()

    from ..config.base import BaseConfig

    defaults = [{"tools": "tflite_converter"}, "_self_"]

    @dataclass
    class Config(BaseConfig):
        defaults: List[Any] = field(default_factory=lambda: defaults)
        tools: Any = MISSING

    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    main()


get_tool_config_objects()["tflite_converter"] = TFLiteConverterConfig
get_tool_run_objects()["tflite_converter"] = run
