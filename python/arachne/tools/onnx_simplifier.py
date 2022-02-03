import itertools
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, List, Optional

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING, DictConfig, OmegaConf

from arachne.utils.global_utils import get_tool_config_objects, get_tool_run_objects
from arachne.utils.model_utils import get_model_spec, load_model_spec, save_model

from ..data import Model


@dataclass
class OnnxSimplifierConfig:
    cli_args: Optional[str] = None


def register_onnx_simplifier_config() -> None:
    cs = ConfigStore.instance()
    group_name = "tools"
    cs.store(
        group=group_name,
        name="onnx_simplifier",
        package="tools.onnx_simplifier",
        node=OnnxSimplifierConfig,
    )


def run(input: Model, cfg: OnnxSimplifierConfig) -> Model:
    idx = itertools.count().__next__()
    filename = f"model_{idx}_simplified.onnx"

    assert input.spec is not None

    cmd = f"python3 -m onnxsim {input.path} {filename} "

    if cfg.cli_args:
        cmd = cmd + str(cfg.cli_args)

    ret = subprocess.run(cmd, shell=True, env={"PATH": os.environ["PATH"]})
    assert ret.returncode == 0

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
