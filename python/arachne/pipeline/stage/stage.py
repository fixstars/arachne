from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from arachne.pipeline.package import Package, PackageInfo

Parameter = Dict[str, Any]


class Stage(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_output_info(input: PackageInfo, params: Parameter) -> Optional[PackageInfo]:
        pass

    @staticmethod
    @abstractmethod
    def extract_parameters(params: Parameter) -> Parameter:
        pass

    @staticmethod
    @abstractmethod
    def process(input: Package, params: Parameter, output_dir: Path) -> Package:
        pass
