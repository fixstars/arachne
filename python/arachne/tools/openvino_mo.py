import itertools
import subprocess
from dataclasses import dataclass, field
from typing import Any, List, Optional

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING, DictConfig, OmegaConf

from arachne.utils import (
    get_model_spec,
    get_tool_config_objects,
    get_tool_run_objects,
    load_model_spec,
    save_model,
)

from ..data import Model


@dataclass
class OpenVINOModelOptConfig:
    cli_args: Optional[str] = None


def register_openvino_mo_config() -> None:
    cs = ConfigStore.instance()
    group_name = "tools"
    cs.store(
        group=group_name,
        name="openvino_mo",
        package="tools.openvino_mo",
        node=OpenVINOModelOptConfig,
    )


def run(input: Model, cfg: OpenVINOModelOptConfig) -> Model:
    idx = itertools.count().__next__()
    assert input.spec is not None
    input_shapes = []
    for inp in input.spec.inputs:
        input_shapes.append(str(inp.shape))

    output_dir = f"openvino_{idx}"
    cmd = [
        "mo",
        "--input_model",
        input.path,
        "--input_shape",
        ",".join(input_shapes),
        "--output_dir",
        output_dir,
    ]

    if cfg.cli_args:
        cmd = cmd + str(cfg.cli_args).split()

    ret = subprocess.run(cmd)
    assert ret.returncode == 0
    return Model(path=output_dir, spec=input.spec)


@hydra.main(config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_model_path = to_absolute_path(cfg.input)
    output_path = to_absolute_path(cfg.output)

    input_model = Model(path=input_model_path, spec=get_model_spec(input_model_path))

    # overwrite model spec if input_spec is specified
    if cfg.input_spec:
        input_model.spec = load_model_spec(to_absolute_path(cfg.input_spec))

    assert input_model.spec is not None
    output_model = run(input=input_model, cfg=cfg.tools.openvino_mo)
    save_model(model=output_model, output_path=output_path)


if __name__ == "__main__":
    register_openvino_mo_config()

    from ..config.base import BaseConfig

    defaults = [{"tools": "openvino_mo"}, "_self_"]

    @dataclass
    class Config(BaseConfig):
        defaults: List[Any] = field(default_factory=lambda: defaults)
        tools: Any = MISSING

    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    main()


get_tool_config_objects()["openvino_mo"] = OpenVINOModelOptConfig
get_tool_run_objects()["openvino_mo"] = run
