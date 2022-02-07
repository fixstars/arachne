import itertools
from dataclasses import dataclass, field
from typing import Any, List, Optional

import hydra
import numpy as np
import onnx
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING, DictConfig, OmegaConf
from onnxsim.onnx_simplifier import simplify

from arachne.utils.global_utils import get_tool_config_objects, get_tool_run_objects
from arachne.utils.model_utils import get_model_spec, load_model_spec, save_model

from ..data import Model


@dataclass
class OnnxSimplifierConfig:
    check_n: int = 3
    skip_fuse_bn: bool = False
    skip_optimization: bool = False
    input_shape: Optional[List[str]] = None
    skip_optimizer: Optional[List[str]] = None
    skip_shape_inference: bool = False
    dynamic_input_shape: bool = False
    input_data_path: Optional[List[str]] = None
    custom_lib: Optional[str] = None


def register_onnx_simplifier_config() -> None:
    cs = ConfigStore.instance()
    group_name = "tools"
    cs.store(
        group=group_name,
        name="onnx_simplifier",
        package="tools.onnx_simplifier",
        node=OnnxSimplifierConfig,
    )


def get_input_shapes_and_tensors_from_args(input_shape, input_data_path):
    # code from onnxsim/__main__.py
    input_shapes = dict()
    if input_shape is not None:
        for x in input_shape:
            if ":" not in x:
                input_shapes[None] = list(map(int, x.split(",")))
            else:
                pieces = x.split(":")
                # for the input name like input:0
                name, shape = ":".join(pieces[:-1]), list(map(int, pieces[-1].split(",")))
                input_shapes.update({name: shape})

    input_data_paths = dict()
    if input_data_path is not None:
        for x in input_data_path:
            pieces = x.split(":")
            name, data = ":".join(pieces[:-1]), pieces[-1]
            input_data_paths.update({name: data})

    input_tensors = dict()
    if len(input_data_paths) > 0 and input_shape is not None:
        for name in input_shapes.keys():
            input_data = np.fromfile(input_data_paths[name], dtype=np.float32)
            input_data = input_data.reshape(input_shapes[name])
            input_tensors.update({name: input_data})
    return input_shapes, input_tensors


def run(input: Model, cfg: OnnxSimplifierConfig) -> Model:
    idx = itertools.count().__next__()
    filename = f"model_{idx}_simplified.onnx"

    assert input.spec is not None

    input_shapes, input_tensors = get_input_shapes_and_tensors_from_args(
        cfg.input_shape, cfg.input_data_path
    )
    model_opt, check_ok = simplify(
        input.path,
        check_n=cfg.check_n,
        perform_optimization=not cfg.skip_optimization,
        skip_fuse_bn=cfg.skip_fuse_bn,
        input_shapes=input_shapes,
        skipped_optimizers=cfg.skip_optimizer,
        skip_shape_inference=cfg.skip_shape_inference,
        input_data=input_tensors,
        dynamic_input_shape=cfg.dynamic_input_shape,
        custom_lib=cfg.custom_lib,
    )
    assert check_ok, "Simplified model validation failed"
    onnx.save(model_opt, filename)

    return Model(filename, spec=get_model_spec(filename))


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
    output_model = run(input=input_model, cfg=cfg.tools.onnx_simplifier)
    save_model(model=output_model, output_path=output_path)


if __name__ == "__main__":
    register_onnx_simplifier_config()

    from ..config.base import BaseConfig

    defaults = [{"tools": "onnx_simplifier"}, "_self_"]

    @dataclass
    class Config(BaseConfig):
        defaults: List[Any] = field(default_factory=lambda: defaults)
        tools: Any = MISSING

    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    main()


get_tool_config_objects()["onnx_simplifier"] = OnnxSimplifierConfig()
get_tool_run_objects()["onnx_simplifier"] = run
