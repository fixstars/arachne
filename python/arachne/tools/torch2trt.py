import itertools
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorrt as trt
import torch
from hydra.utils import to_absolute_path
from torch2trt import DEFAULT_CALIBRATION_ALGORITHM
from torch2trt import torch2trt as run_torch2trt

from arachne.tools.factory import (
    ToolBase,
    ToolConfigBase,
    ToolConfigFactory,
    ToolFactory,
)
from arachne.utils.torch_utils import get_torch_dtype_from_string

from ..data import Model

_FACTORY_KEY = "torch2trt"


@ToolConfigFactory.register(_FACTORY_KEY)
@dataclass
class Torch2TRTConfig(ToolConfigBase):
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


@ToolFactory.register(_FACTORY_KEY)
class Torch2TRT(ToolBase):
    @staticmethod
    def run(input: Model, cfg: Torch2TRTConfig) -> Model:
        idx = itertools.count().__next__()
        filename = f"model_{idx}_trt.pth"

        model = torch.load(input.path).eval().cuda()

        assert input.spec is not None
        args = []
        for inp in list(input.spec.inputs):
            x = torch.randn(
                *inp.shape,
                dtype=get_torch_dtype_from_string(inp.dtype),
                device=torch.device("cuda"),
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
