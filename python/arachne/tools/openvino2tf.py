import itertools
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, List, Optional

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING, DictConfig, OmegaConf

from arachne.utils import get_model_spec, save_model

from ..data import Model


@dataclass
class OpenVINO2TFConfig:
    cli_args: Optional[str] = None


def register_openvino_mo_config() -> None:
    cs = ConfigStore.instance()
    group_name = "tools"
    cs.store(
        group=group_name,
        name="openvino2tf",
        package="tools.openvino2tf",
        node=OpenVINO2TFConfig,
    )


def _find_openvino_xml_file(dir: str) -> Optional[str]:
    for f in os.listdir(dir):
        if f.endswith(".xml"):
            return dir + "/" + f
    return None


def run(input: Model, cfg: OpenVINO2TFConfig) -> Model:
    idx = itertools.count().__next__()
    assert input.spec is not None
    input_shapes = []
    for inp in input.spec.inputs:
        input_shapes.append(str(inp.shape))

    output_dir = f"openvino2tf-{idx}-saved_model"

    if input.path.endswith(".xml"):
        model_path = input.path
    else:
        model_path = _find_openvino_xml_file(input.path)
    assert model_path is not None
    cmd = [
        "openvino2tensorflow",
        "--model_path",
        model_path,
        "--model_output_path",
        output_dir,
        "--output_saved_model",
    ]

    if cfg.cli_args:
        cmd = cmd + str(cfg.cli_args).split()

    ret = subprocess.run(cmd)
    assert ret.returncode == 0
    return Model(path=output_dir, spec=get_model_spec(output_dir))


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
    output_model = run(input=input_model, cfg=cfg.tools.openvino2tf)
    save_model(model=output_model, output_path=output_path, cfg=cfg)


if __name__ == "__main__":
    register_openvino_mo_config()

    from ..config.base import BaseConfig

    defaults = [{"tools": "openvino2tf"}, "_self_"]

    @dataclass
    class Config(BaseConfig):
        defaults: List[Any] = field(default_factory=lambda: defaults)
        tools: Any = MISSING

    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    main()
