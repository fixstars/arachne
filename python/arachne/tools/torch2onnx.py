import itertools
from dataclasses import dataclass, field
from typing import Any, List, Optional

import hydra
import torch
import torch.onnx
import torch.onnx.symbolic_helper
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING, DictConfig, OmegaConf

from arachne.utils import (
    get_model_spec,
    get_tool_config_objects,
    get_tool_run_objects,
    get_torch_dtype_from_string,
    save_model,
)

from ..data import Model


@dataclass
class Torch2ONNXConfig:
    export_params: bool = True
    verbose: bool = False
    training: int = torch.onnx.TrainingMode.EVAL.value
    operator_export_type: Optional[int] = None
    opset_version: int = torch.onnx.symbolic_helper._default_onnx_opset_version
    do_constant_folding: bool = False
    dynamic_axes: Optional[Any] = None
    keep_initializers_as_inputs: Optional[bool] = None
    custom_opsets: Optional[Any] = None


def register_torch2onnx_config() -> None:
    cs = ConfigStore.instance()
    group_name = "tools"
    cs.store(
        group=group_name,
        name="torch2onnx",
        package="tools.torch2onnx",
        node=Torch2ONNXConfig,
    )


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
    output_model = run(input=input_model, cfg=cfg.tools.torch2onnx)
    save_model(model=output_model, output_path=output_path, cfg=cfg)


if __name__ == "__main__":
    register_torch2onnx_config()

    from ..config.base import BaseConfig

    defaults = [{"tools": "torch2onnx"}, "_self_"]

    @dataclass
    class Config(BaseConfig):
        defaults: List[Any] = field(default_factory=lambda: defaults)
        tools: Any = MISSING

    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    main()


get_tool_config_objects()["torch2onnx"] = Torch2ONNXConfig
get_tool_run_objects()["torch2onnx"] = run
