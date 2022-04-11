import itertools
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from hydra.utils import to_absolute_path
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from arachne.tools.factory import (
    ToolBase,
    ToolConfigBase,
    ToolConfigFactory,
    ToolFactory,
)

from ..data import Model, ModelFormat, TensorSpec

_FACTORY_KEY = "tftrt"


@ToolConfigFactory.register(_FACTORY_KEY)
@dataclass
class TFTRTConfig(ToolConfigBase):
    """This is a class for configuring the behavior of the TF-TRT.

    Attributes:
        max_workspace_size_bytes (int): The maximum GPU temporary memory which the TensorRT engine can use at execution time. Default value is 1GB.

        precision_mode (str):  This is one of "FP32", "FP16", or "INT8". Default value is False.

        minimum_segment_size (int): This is the minimum number of nodes required for a subgraph to be replaced by the TRT operator. Default value is 3.

        maximum_cached_engines (int): This is the maximum number of cached TensorRT engines in TensorFlow for each TensorRT subgraph. Default value is 1.

        use_calibration (bool): Whether to use calibration. This argument is ignored if precision_mode is not "INT8". Default value is True.

        allow_build_at_runtime (bool): Whether to allow building TensorRT engines during runtime. Default value is True.

        representative_dataset (:obj:`str`, optional): A path to calibration dataset (a npy file). Default value is None.

    """

    max_workspace_size_bytes: int = 1 << 30
    precision_mode: str = "FP32"
    minimum_segment_size: int = 3
    maximum_cached_engines: int = 1
    use_calibration: bool = True
    allow_build_at_runtime: bool = True
    representative_dataset: Optional[str] = None


@ToolFactory.register(_FACTORY_KEY)
class TFTRT(ToolBase):
    """This is a runner class for executing the TF-TRT."""

    @staticmethod
    def run(input: Model, cfg: TFTRTConfig) -> Model:
        """
        The run method is a static method that executes TF-TRT for an input model.

        Args:
            input (Model): An input model.
            cfg (TFTRTConfig): A config object.
        Returns:
            Model: A Tensorflow Model optimized for TensorRT.
        """
        idx = itertools.count().__next__()
        params = trt.DEFAULT_TRT_CONVERSION_PARAMS
        params = params._replace(max_workspace_size_bytes=cfg.max_workspace_size_bytes)
        params = params._replace(precision_mode=cfg.precision_mode)
        params = params._replace(minimum_segment_size=cfg.minimum_segment_size)
        params = params._replace(maximum_cached_engines=cfg.maximum_cached_engines)
        params = params._replace(use_calibration=cfg.use_calibration)
        params = params._replace(allow_build_at_runtime=cfg.allow_build_at_runtime)

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
        return Model(
            path=output_saved_model_dir, format=ModelFormat.TF_SAVED_MODEL, spec=input.spec
        )
