import itertools
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

_FACTORY_KEY = "onnx_tf"


@ToolConfigFactory.register(_FACTORY_KEY)
@dataclass
class ONNXTfConfig(ToolConfigBase):
    cli_args: Optional[str] = None


@ToolFactory.register(_FACTORY_KEY)
class ONNXTf(ToolBase):
    @staticmethod
    def run(input: Model, cfg: ONNXTfConfig) -> Model:
        idx = itertools.count().__next__()
        output_dir = f"model_{idx}-saved_model"
        assert input.spec is not None

        cmd = [
            "onnx-tf",
            "convert",
            "-i",
            input.path,
            "-o",
            output_dir,
        ]

        if cfg.cli_args:
            cmd = cmd + str(cfg.cli_args).split()

        ret = subprocess.run(cmd)
        assert ret.returncode == 0
        return Model(path=output_dir, spec=get_model_spec(output_dir))
