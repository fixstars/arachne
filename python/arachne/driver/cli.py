from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, List

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING, DictConfig, OmegaConf

from arachne.data import Model
from arachne.tools import ToolFactory
from arachne.utils.model_utils import get_model_spec, load_model_spec, save_model

logger = getLogger(__name__)


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:

    tools = list(cfg.tools.keys())
    assert len(tools) == 1, "Error: you must specify only one tool"
    tool = tools[0]

    logger.info(OmegaConf.to_yaml(cfg))

    input_model_path = to_absolute_path(cfg.input)
    output_path = to_absolute_path(cfg.output)

    input_model = Model(path=input_model_path, spec=get_model_spec(input_model_path))

    # overwrite model spec if input_spec is specified
    if cfg.input_spec:
        input_model.spec = load_model_spec(to_absolute_path(cfg.input_spec))

    assert input_model.spec is not None

    output_model = ToolFactory.get(tool).run(input=input_model, cfg=cfg.tools.get(tool))
    save_model(model=output_model, output_path=output_path, tvm_cfg=cfg.tools.get(tool))


if __name__ == "__main__":
    from ..config.base import BaseConfig

    defaults = [{"override hydra/job_logging": "custom"}, "_self_"]

    @dataclass
    class Config(BaseConfig):
        defaults: List[Any] = field(default_factory=lambda: defaults)
        tools: Any = MISSING

    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    main()
