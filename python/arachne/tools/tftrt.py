import itertools
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import tensorflow as tf
from hydra.utils import to_absolute_path
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from arachne.tools.factory import (
    ToolBase,
    ToolConfigBase,
    ToolConfigFactory,
    ToolFactory,
)

from ..data import Model, TensorSpec

_FACTORY_KEY = "tftrt"


@ToolConfigFactory.register(_FACTORY_KEY)
@dataclass
class TFTRTConfig(ToolConfigBase):
    max_workspace_size_bytes: int = 1 << 30
    precision_mode: str = "FP32"
    minimum_segment_size: int = 3
    maximum_cached_engines: int = 1
    use_calibration: bool = True
    is_dynamic_op: bool = True
    max_batch_size: int = 1
    allow_build_at_runtime: bool = True
    representative_dataset: Optional[str] = None


@ToolFactory.register(_FACTORY_KEY)
class TFTRT(ToolBase):
    @staticmethod
    def run(input: Model, cfg: TFTRTConfig) -> Model:
        idx = itertools.count().__next__()
        params = trt.DEFAULT_TRT_CONVERSION_PARAMS
        params = params._replace(max_workspace_size_bytes=cfg.max_workspace_size_bytes)
        params = params._replace(precision_mode=cfg.precision_mode)
        params = params._replace(minimum_segment_size=cfg.minimum_segment_size)
        params = params._replace(maximum_cached_engines=cfg.maximum_cached_engines)
        params = params._replace(use_calibration=cfg.use_calibration)
        params = params._replace(allow_build_at_runtime=cfg.allow_build_at_runtime)

        tf_version = tf.__version__.split(".")
        tf_major_version = int(tf_version[0])
        tf_minor_version = int(tf_version[1])
        if tf_major_version == 2 and tf_minor_version < 4:
            params = params._replace(is_dynamic_op=cfg.is_dynamic_op)
            params = params._replace(max_batch_size=cfg.max_batch_size)

        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=input.path, conversion_params=params
        )

        calibration_input_fn = None
        if cfg.precision_mode == "INT8":
            # TODO support multiple inputs
            datasets: List[np.ndarray]
            if cfg.representative_dataset is not None:
                datasets = np.load(to_absolute_path(cfg.representative_dataset))  # type: ignore
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

            calibration_input_fn = representative_dataset

        converter.convert(calibration_input_fn=calibration_input_fn)
        output_saved_model_dir = f"tftrt-{idx}-saved_model"
        converter.save(output_saved_model_dir)
        return Model(path=output_saved_model_dir, spec=input.spec)
