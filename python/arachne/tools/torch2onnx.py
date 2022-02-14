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
from arachne.utils.model_utils import get_model_spec
from arachne.utils.torch_utils import get_torch_dtype_from_string

from ..data import Model

_FACTORY_KEY = "torch2onnx"


@ToolConfigFactory.register(_FACTORY_KEY)
@dataclass
class Torch2ONNXConfig(ToolConfigBase):
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
    @staticmethod
    def run(input: Model, cfg: Torch2ONNXConfig) -> Model:
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
            training=cfg.training,
            operator_export_type=cfg.operator_export_type,
            opset_version=cfg.opset_version,
            do_constant_folding=cfg.do_constant_folding,
            dynamic_axes=cfg.dynamic_axes,
            custom_opsets=cfg.custom_opsets,
        )
        return Model(filename, spec=get_model_spec(filename))
