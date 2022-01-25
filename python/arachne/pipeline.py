from dataclasses import dataclass, field
from typing import Any, Dict, List

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline, node
from kedro.runner import SequentialRunner
from omegaconf import MISSING, DictConfig, OmegaConf

from .config.base import BaseConfig
from .data import Model
from .tools.openvino2tf import register_openvino2tf_config
from .tools.openvino_mo import register_openvino_mo_config
from .tools.tflite_converter import register_tflite_converter_config
from .tools.tftrt import register_tftrt_config
from .tools.torch2onnx import register_torch2onnx_config
from .tools.torch2trt import register_torch2trt_config
from .tools.tvm import register_tvm_config
from .utils.global_utils import get_tool_config_objects, get_tool_run_objects
from .utils.model_utils import get_model_spec, load_model_spec, save_model


def register_tools_config():
    register_openvino2tf_config()
    register_openvino_mo_config()
    register_tflite_converter_config()
    register_tftrt_config()
    register_torch2onnx_config()
    register_torch2trt_config()
    register_tvm_config()


def get_all_tools() -> List[str]:
    return list(get_tool_config_objects().keys())


@dataclass
class PipelineConfig(BaseConfig):
    tools: Any = MISSING
    pipeline: List[str] = MISSING


def get_default_tool_configs(tools: List[str]) -> Dict:
    all_tools = get_all_tools()
    config = {}
    for t in tools:
        if t not in all_tools:
            assert False, f"Not supported tool ({t}) yet"
        config[t] = get_tool_config_objects()[t]()

    return config


def run(input: Model, cfg: PipelineConfig) -> Model:

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
    run_objs = get_tool_run_objects()
    for idx, tool in enumerate(cfg.pipeline):
        run = run_objs[tool]
        config = "tools." + tool + "." + str(idx) + ".config"
        output = "tools." + tool + "." + str(idx) + ".output"
        tool_inputs = {"input": prev_output, "cfg": config}
        task = node(run, inputs=tool_inputs, outputs=output)
        prev_output = output
        pipeline_tmp.append(task)

    pipeline = Pipeline(pipeline_tmp)

    # Create a runner to run the pipeline
    runner = SequentialRunner()

    # Run the pipeline
    runner.run(pipeline, data_catalog)

    return data_catalog.load(prev_output)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_model_path = to_absolute_path(cfg.input)
    output_path = to_absolute_path(cfg.output)

    input_model = Model(path=input_model_path, spec=get_model_spec(input_model_path))

    # overwrite model spec if input_spec is specified
    if cfg.input_spec:
        input_model.spec = load_model_spec(to_absolute_path(cfg.input_spec))

    assert input_model.spec is not None

    output_model = run(input_model, cfg)  # type: ignore

    save_model(model=output_model, output_path=output_path, tvm_cfg=cfg.tools.tvm)


if __name__ == "__main__":
    register_tools_config()

    defaults = [{"tools": get_all_tools()}, "_self_"]

    @dataclass
    class PipelineCLIConfig(PipelineConfig):
        defaults: List[Any] = field(default_factory=lambda: defaults)

    cs = ConfigStore.instance()
    cs.store(name="config", node=PipelineCLIConfig)
    main()
