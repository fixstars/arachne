from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, List

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING, DictConfig, OmegaConf

from arachne.data import Model
from arachne.tools import ToolFactory
from arachne.utils.model_utils import (
    init_from_dir,
    init_from_file,
    load_model_spec,
    save_model,
)

logger = getLogger(__name__)


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    This is a main function for `arachne.driver.cli`.
    """

    logger.info(OmegaConf.to_yaml(cfg))

    # Check the specified tool is valid
    tools = list(cfg.tools.keys())

    try:
        assert len(tools) == 1
    except AssertionError as err:
        logger.exception("You must specify only one tool")
        raise err

    tool = tools[0]

    # Setup the input DNN model

    if not cfg.model_file and not cfg.model_dir:
        raise RuntimeError("User must specify either model_file or model_dir.")
    if cfg.model_file and cfg.model_dir:
        raise RuntimeError("User must specify either model_file or model_dir.")

    input_model: Model
    if cfg.model_file:
        input_model = init_from_file(to_absolute_path(cfg.model_file))
    else:
        input_model = init_from_dir(to_absolute_path(cfg.model_dir))

    if cfg.model_spec_file:
        # if a YAML file describing the model specification is provided, overwrite input_model.spec
        input_model.spec = load_model_spec(to_absolute_path(cfg.model_spec_file))

    output_model = ToolFactory.get(tool).run(input=input_model, cfg=cfg.tools.get(tool))
    save_model(
        model=output_model,
        output_path=to_absolute_path(cfg.output_path),
        tvm_cfg=cfg.tools.get(tool),
    )


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
