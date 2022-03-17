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

from ..data import Model

_FACTORY_KEY = "openvino_mo"


@ToolConfigFactory.register(_FACTORY_KEY)
@dataclass
class OpenVINOModelOptConfig(ToolConfigBase):
    """The OpenVINO Model Optimizer has many options, so that we decided to provide only one string parameter that will be passed to the optimizer.
    To understand what options are available, run `mo --help`.
    """

    cli_args: Optional[str] = None


@ToolFactory.register(_FACTORY_KEY)
class OpenVINOModelOptimizer(ToolBase):
    """This is a runner class for executing the OpenVINO Model Optmizier."""

    @staticmethod
    def run(input: Model, cfg: OpenVINOModelOptConfig) -> Model:
        """
        The run method is a static method that executes mo.py for an input model.

        Args:
            input (Model): An input model.
            cfg (OpenVINOModelOptConfig): A config object.
        Returns:
            Model: A OpenVINO IR.
        """
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
