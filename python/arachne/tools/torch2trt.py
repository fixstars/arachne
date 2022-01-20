import itertools
from dataclasses import dataclass, field
from typing import Any, List, Optional

import hydra
import numpy as np
import tensorrt as trt
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING, DictConfig, OmegaConf
from torch2trt import DEFAULT_CALIBRATION_ALGORITHM
from torch2trt import torch2trt as run_torch2trt

from arachne.utils import (
    get_model_spec,
    get_tool_config_objects,
    get_tool_run_objects,
    get_torch_dtype_from_string,
    load_model_spec,
    save_model,
)

from ..data import Model


@dataclass
class Torch2TRTConfig:
    max_batch_size: int = 1
    fp16_mode: bool = False
    max_workspace_size: int = 1 << 25
    strict_type_constraints: bool = False
    keep_network: bool = True
    int8_mode: bool = False
    int8_calib_dataset: Optional[str] = None
    int8_calib_algorithm: str = "DEFAULT"
    int8_calib_batch_size: int = 1
    use_onnx: bool = False


def register_torch2trt_config() -> None:
    cs = ConfigStore.instance()
    group_name = "tools"
    cs.store(
        group=group_name,
        name="torch2trt",
        package="tools.torch2trt",
        node=Torch2TRTConfig,
    )


def run(input: Model, cfg: Torch2TRTConfig) -> Model:
    idx = itertools.count().__next__()
    filename = f"model_{idx}_trt.pth"

    model = torch.load(input.path).eval().cuda()

    assert input.spec is not None
    args = []
    for inp in list(input.spec.inputs):
        x = torch.randn(
            *inp.shape, dtype=get_torch_dtype_from_string(inp.dtype), device=torch.device("cuda")
        )
        args.append(x)

    dataset = None
    algo = None
    if cfg.int8_mode and cfg.int8_calib_dataset:
        datasets = np.load(to_absolute_path(cfg.int8_calib_dataset))  # type: ignore

        class CalibDataset:
            def __init__(self, dataset):
                self.dataset = dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                image = self.dataset[idx]
                x = torch.from_numpy(image).clone()  # type: ignore
                x = x.to("cuda")
                return [x[0]]

        dataset = CalibDataset(datasets)

    if cfg.int8_mode and cfg.int8_calib_algorithm != "DEFAULT":
        if cfg.int8_calib_algorithm == "ENTROPY_CALIBRATION_2":
            algo = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2  # type: ignore
        elif cfg.int8_calib_algorithm == "ENTROPY_CALIBRATION":
            algo = trt.CalibrationAlgoType.ENTROPY_CALIBRATION  # type: ignore
        elif cfg.int8_calib_algorithm == "LEGACY_CALIBRATION":
            assert (
                False
            ), "LEGACY_CALIBRATION caused a SEGV for resnet-18, so we restricted this algo"
        elif cfg.int8_calib_algorithm == "MINMAX_CALIBRATION":
            algo = trt.CalibrationAlgoType.MINMAX_CALIBRATION  # type: ignore
        else:
            algo = DEFAULT_CALIBRATION_ALGORITHM

    model_trt = run_torch2trt(
        model,
        args,
        max_batch_size=cfg.max_batch_size,
        fp16_mode=cfg.fp16_mode,
        max_workspace_size=cfg.max_workspace_size,
        strict_type_constraints=cfg.strict_type_constraints,
        keep_network=cfg.keep_network,
        int8_mode=cfg.int8_mode,
        int8_calib_dataset=dataset,
        int8_calib_algorithm=algo,
        int8_calib_batch_size=cfg.int8_calib_batch_size,
        use_onnx=cfg.use_onnx,
    )

    torch.save(model_trt.state_dict(), filename)
    return Model(filename, spec=input.spec)


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_model_path = to_absolute_path(cfg.input)
    output_path = to_absolute_path(cfg.output)

    input_model = Model(path=input_model_path, spec=get_model_spec(input_model_path))

    # overwrite model spec if input_spec is specified
    if cfg.input_spec:
        input_model.spec = load_model_spec(to_absolute_path(cfg.input_spec))

    assert input_model.spec is not None
    output_model = run(input=input_model, cfg=cfg.tools.torch2trt)
    save_model(model=output_model, output_path=output_path)


if __name__ == "__main__":
    register_torch2trt_config()

    from ..config.base import BaseConfig

    defaults = [{"tools": "torch2trt"}, "_self_"]

    @dataclass
    class Config(BaseConfig):
        defaults: List[Any] = field(default_factory=lambda: defaults)
        tools: Any = MISSING

    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    main()


get_tool_config_objects()["torch2trt"] = Torch2TRTConfig
get_tool_run_objects()["torch2trt"] = run
