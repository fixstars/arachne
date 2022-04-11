from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class BaseConfig:
    """This is a base configuration class for arachne.driver.cli and arachne.driver.pipeline.

    Attributes:
        model_file (:obj:`str`, optional): A path to a input model file.
        model_dir (:obj:`str`, optional): A path to a input model directory.
        model_spec_file (:obj:`str`, optional): A path for a YAML file showing the tensor specification of the input model. Default value is None.
        output_file (str): A path for saveing output model.
    """

    model_file: Optional[str] = None
    model_dir: Optional[str] = None
    model_spec_file: Optional[str] = None
    output_path: str = MISSING
