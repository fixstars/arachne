from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class BaseConfig:
    """This is a base configuration class for arachne.driver.cli and arachne.driver.pipeline.

    Attributes:
        input (str): A path for input model.
        input_spec (:obj:`str`, optional): A path for a YAML file showing the tensor specification of the input model. Default value is None.
        output (str): A path for output model.
    """
    input: str = MISSING
    input_spec: Optional[str] = None
    output: str = MISSING
