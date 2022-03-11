import itertools
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

from arachne.tools.factory import (
    ToolBase,
    ToolConfigBase,
    ToolConfigFactory,
    ToolFactory,
)
from arachne.utils.model_utils import get_model_spec

from ..data import Model

_FACTORY_KEY = "openvino2tf"


@ToolConfigFactory.register(_FACTORY_KEY)
@dataclass
class OpenVINO2TFConfig(ToolConfigBase):
    """The openvino2tensorflow has many options, so that we decided to provide only one string parameter that will be passed to the tool.
    To understand what options are available, run `openvino2tensorflow --help`.
    """
    cli_args: Optional[str] = None


def _find_openvino_xml_file(dir: str) -> Optional[str]:
    for f in os.listdir(dir):
        if f.endswith(".xml"):
            return dir + "/" + f
    return None


@ToolFactory.register(_FACTORY_KEY)
class OpenVINO2TF(ToolBase):
    """This is a runner class for executing the openvino2tensorflow.
    """
    @staticmethod
    def run(input: Model, cfg: OpenVINO2TFConfig) -> Model:
        """
        The run method is a static method that executes openvino2tensorflow for an input model.

        Args:
            input (Model): An input model.
            cfg (OpenVINO2TFConfig): A config object.
        Returns:
            Model: A Tensorflow Model.
        """
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
