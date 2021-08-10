from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from arachne.pipeline.package import Package, PackageInfo

Parameter = Dict[str, Any]


class Stage(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def get_output_info(cls, input: PackageInfo, params: Parameter) -> Optional[PackageInfo]:
        pass

    @classmethod
    @abstractmethod
    def extract_parameters(cls, params: Parameter) -> Parameter:
        pass

    @classmethod
    @abstractmethod
    def process(cls, input: Package, params: Parameter, output_dir: Path) -> Package:
        pass
