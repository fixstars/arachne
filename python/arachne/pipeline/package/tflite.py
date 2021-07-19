from pathlib import Path
from typing import List

import attr
from arachne.types import QType

from .package import Package, PackageInfo


@attr.s(auto_attribs=True, frozen=True)
class TfLitePackageInfo(PackageInfo):
    qtype: QType
    for_edgetpu: bool


@attr.s(auto_attribs=True, frozen=True)
class TfLitePackage(TfLitePackageInfo, Package):
    model_file: Path

    @property
    def files(self) -> List[Path]:
        return [self.model_file]
