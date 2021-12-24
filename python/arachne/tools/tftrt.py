import itertools
from dataclasses import dataclass, field
from typing import Any, List, Optional

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING, DictConfig, OmegaConf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from arachne.utils import get_model_spec, save_model

from ..data import Model, TensorSpec


@dataclass
class TFTRTConfig:
    max_workspace_size_bytes: int = 1 << 30
    precision_mode: str = "FP32"
    minimum_segment_size: int = 3
    maximum_cached_engines: int = 1
    use_calibration: bool = True
    is_dynamic_op: bool = True
    max_batch_size: int = 1
    allow_build_at_runtime: bool = True
    representative_dataset: Optional[str] = None


def register_tftrt_config() -> None:
    cs = ConfigStore.instance()
    group_name = "tools"
    cs.store(
        group=group_name,
        name="tftrt",
        package="tools.tftrt",
        node=TFTRTConfig,
    )


def run(input: Model, cfg: TFTRTConfig) -> Model:
    idx = itertools.count().__next__()
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    params = params._replace(max_workspace_size_bytes=cfg.max_workspace_size_bytes)
    params = params._replace(precision_mode=cfg.precision_mode)
    params = params._replace(minimum_segment_size=cfg.minimum_segment_size)
    params = params._replace(maximum_cached_engines=cfg.maximum_cached_engines)
    params = params._replace(use_calibration=cfg.use_calibration)
    params = params._replace(is_dynamic_op=cfg.is_dynamic_op)
    params = params._replace(max_batch_size=cfg.max_batch_size)
    params = params._replace(allow_build_at_runtime=cfg.allow_build_at_runtime)

    converter = trt.TrtGraphConverterV2(input_saved_model_dir=input.path, conversion_params=params)

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


@hydra.main(config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_model_path = to_absolute_path(cfg.input)
    output_path = to_absolute_path(cfg.output)

    input_model = Model(path=input_model_path, spec=get_model_spec(input_model_path))

    # overwrite model spec if input_spec is specified
    if cfg.input_spec:
        input_model.spec = OmegaConf.load(to_absolute_path(cfg.input_spec))  # type: ignore

    assert input_model.spec is not None
    output_model = run(input=input_model, cfg=cfg.tools.tftrt)
    save_model(model=output_model, output_path=output_path, cfg=cfg)


if __name__ == "__main__":
    register_tftrt_config()

    from ..config.base import BaseConfig

    defaults = [{"tools": "tftrt"}, "_self_"]

    @dataclass
    class Config(BaseConfig):
        defaults: List[Any] = field(default_factory=lambda: defaults)
        tools: Any = MISSING

    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    main()
