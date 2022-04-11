import itertools
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.onnx
import torch.onnx.symbolic_helper

from arachne.tools.factory import (
    ToolBase,
    ToolConfigBase,
    ToolConfigFactory,
    ToolFactory,
)
from arachne.utils.model_utils import init_from_file
from arachne.utils.torch_utils import get_torch_dtype_from_string

from ..data import Model

_FACTORY_KEY = "torch2onnx"


@ToolConfigFactory.register(_FACTORY_KEY)
@dataclass
class Torch2ONNXConfig(ToolConfigBase):
    """This class aims to covers the parameters of the torch.onnx.export().
    If you want to know the details, please refer the original docstring of torch.onnx.export().
    https://pytorch.org/docs/stable/onnx.html#torch.onnx.export

    Attributes:
        export_params  (bool): If specified, all parameters will be exported. Default value is True.

        verbose (bool): if sepecified, torch.onnx.export() will dump a debug message.

        training (int): Changes the export behavior. Possible values are TrainingMode.EVAL, TrainingMode.PRESERVE, or TrainingMode.TRAINING.
        Default value is TrainingMode.EVAL.

        operator_export_type (:obj:`int`, optional): Changes the type of exported operators. Possible values are OperatorExportTypes.ONNX, OperatorExportTypes.ONNX_ATEN, or OperatorExportTypes.ONNX_ATEN_FALLBACK. Default value is None.

        opset_version (int): Changes the opset version. Default value is 9.

        do_constant_folding (bool): If True, the constant-folding optimization is applied to the model during export. Default value is False.

        dynamix_axes (:obj:`Any`, optional): A dictionary to specify dynamic axes of input/output. Default value is False.

        keep_initializers_as_inputs (:obj:`bool`, optional): Changes whether the initializers will be added as inputs to the graph or not. Default is None.

        custom_opsets (:obj:`Any`, optional): A dictionary to indicate custom opset domain and version at export.
    """

    export_params: bool = True
    verbose: bool = False
    training: int = torch.onnx.TrainingMode.EVAL.value
    operator_export_type: Optional[int] = None
    opset_version: int = torch.onnx.symbolic_helper._default_onnx_opset_version
    do_constant_folding: bool = False
    dynamic_axes: Optional[Any] = None
    keep_initializers_as_inputs: Optional[bool] = None
    custom_opsets: Optional[Any] = None


@ToolFactory.register(_FACTORY_KEY)
class Torch2ONNX(ToolBase):
    """This is a runner class for executing torch.onnx.export()."""

    @staticmethod
    def run(input: Model, cfg: Torch2ONNXConfig) -> Model:
        """
        The run method is a static method that executes torch.onnx.export() for an input model.

        Args:
            input (Model): An input model.
            cfg (Torch2ONNXConfit): A config object.
        Returns:
            Model: A exported ONNX model.
        """
        idx = itertools.count().__next__()
        filename = f"model_{idx}.onnx"

        assert input.spec is not None

        model = torch.load(input.path)

        args = []
        for inp in list(input.spec.inputs):
            x = torch.randn(*inp.shape, dtype=get_torch_dtype_from_string(inp.dtype))
            args.append(x)
        args = tuple(args)

        torch.onnx.export(
            model=model,
            args=args,
            f=filename,
            export_params=cfg.export_params,
            verbose=cfg.verbose,
            training=cfg.training,  # type: ignore
            operator_export_type=cfg.operator_export_type,
            opset_version=cfg.opset_version,
            do_constant_folding=cfg.do_constant_folding,
            dynamic_axes=cfg.dynamic_axes,
            custom_opsets=cfg.custom_opsets,
        )

        return init_from_file(filename)
