from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict, List

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline, node
from kedro.runner import SequentialRunner
from omegaconf import MISSING, DictConfig, OmegaConf

from arachne.config.base import BaseConfig
from arachne.data import Model
from arachne.tools import ToolConfigFactory, ToolFactory
from arachne.utils.model_utils import (
    init_from_dir,
    init_from_file,
    load_model_spec,
    save_model,
)

logger = getLogger(__name__)


@dataclass
class PipelineConfig(BaseConfig):
    """
    This is a configuration class for pipeline.

    Attributes:
        tools (Any): Tool specific configurations. This will be overwritten by the value of the pipeline option.
        pipeline (List[str]): Specifies tools to be applied in series.
    """

    tools: Any = MISSING
    pipeline: List[str] = MISSING


def get_default_tool_configs(tools: List[str]) -> Dict:
    all_tools = ToolFactory.list()
    config = {}
    for t in tools:
        if t not in all_tools:
            assert False, f"Not supported tool ({t}) yet"
        config[t] = ToolConfigFactory.get(t)

    return config


def run(input: Model, cfg: PipelineConfig) -> Model:
    """
    This function constructs and applies a pipeline defined by the config object to an input model.

    Args:
        input (Model): An input model.
        cfg (PipelineConfig): A config object.
    Returns:
        Model: An output model.
    """

    # Preprare DataCatalog
    data_catalog = DataCatalog()
    data_catalog.add("root_input", MemoryDataSet(data=input))

    # setup catalogs for each tool configs and outputs
    for idx, tool in enumerate(cfg.pipeline):
        config = "tools." + tool + "." + str(idx) + ".config"
        output = "tools." + tool + "." + str(idx) + ".output"
        data_catalog.add(config, MemoryDataSet(data=cfg.tools[tool]))
        data_catalog.add(output, MemoryDataSet())

    # Construct pipeline
    pipeline_tmp = []
    prev_output = "root_input"
    for idx, tool in enumerate(cfg.pipeline):
        t = ToolFactory.get(tool)
        config = "tools." + tool + "." + str(idx) + ".config"
        output = "tools." + tool + "." + str(idx) + ".output"
        tool_inputs = {"input": prev_output, "cfg": config}
        task = node(t.run, inputs=tool_inputs, outputs=output)
        prev_output = output
        pipeline_tmp.append(task)

    pipeline = Pipeline(pipeline_tmp)

    # Create a runner to run the pipeline
    runner = SequentialRunner()

    # Run the pipeline
    runner.run(pipeline, data_catalog)

    return data_catalog.load(prev_output)


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    This is a main function for `arachne.driver.pipeline`.
    """

    try:
        assert len(list(cfg.pipeline)) > 0
    except AssertionError as err:
        logger.exception("You must specify one tool at least")
        raise err

    logger.info(OmegaConf.to_yaml(cfg))

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

    output_model = run(input_model, cfg)  # type: ignore
    save_model(
        model=output_model, output_path=to_absolute_path(cfg.output_path), tvm_cfg=cfg.tools.tvm
    )


if __name__ == "__main__":
    defaults = [{"tools": ToolFactory.list()}, {"override hydra/job_logging": "custom"}, "_self_"]

    @dataclass
    class PipelineCLIConfig(PipelineConfig):
        defaults: List[Any] = field(default_factory=lambda: defaults)

    cs = ConfigStore.instance()
    cs.store(name="config", node=PipelineCLIConfig)
    main()
