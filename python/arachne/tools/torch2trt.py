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
    """This is a class for configuring the behavior of the torch2trt.

    Attributes:
        max_batch_size  (int): This is the max size for the input batch. Default value is 1.

        fp16_mode (bool): To convert a model with fp16_mode=True allows the TensorRT optimizer to select layers with fp16 precision. Default value is False.

        max_workspace_size (int): The maximum GPU temporary memory which the TensorRT engine can use at execution time

        strict_type_constraints (bool): Enables strict type constraints. But, this flag is deprecated. Default value is False.

        keep_network (bool): Whether to hold the optimized network. Default value is True.

        int8_mode (bool): Enables INT8 precision. Default value is False.

        int8_calib_dataset (:obj:`str`, optional): A path to calibration dataset (*.npy). Default value is None.

        int8_calib_algorithm (str): To override the default calibration algorithm. Possible values are "DEFAULT", "ENTROPY_CALIBRATION", "ENTROPY_CALIBRATION_2", or "MINMAX_CALIBRATION". Default value is "DEFAULT".

        int8_calib_batch_size (int): To set the calibration batch size, you can set the this parameter. Default value is 1.

        use_onnx (bool): If you set use_onnx=True, this will perform conversion on the module by exporting the model using PyTorch's JIT tracer, and parsing with TensorRT's ONNX parser. Default value is False.
    """

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
    """This is a runner class for executing torch2trt."""

    @staticmethod
    def run(input: Model, cfg: Torch2TRTConfig) -> Model:
        """
        The run method is a static method that executes torch2trt() for an input model.

        Args:
            input (Model): An input model.
            cfg (Torch2TRTConfig): A config object.
        Returns:
            Model: An optimized model.
        """
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
