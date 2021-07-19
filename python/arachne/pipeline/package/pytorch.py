from pathlib import Path
from typing import List

import attr

from .package import Package, PackageInfo


@attr.s(auto_attribs=True, frozen=True)
class PyTorchPackageInfo(PackageInfo):
    quantizable: bool


@attr.s(auto_attribs=True, frozen=True)
class PyTorchPackage(PyTorchPackageInfo, Package):
    model_file: Path

    @property
    def files(self) -> List[Path]:
        return [self.model_file]
