from pathlib import Path
from typing import List

import attr

from arachne.runtime.qtype import QType

from .package import Package, PackageInfo


@attr.s(auto_attribs=True, frozen=True)
class TFLitePackageInfo(PackageInfo):
    qtype: QType
    for_edgetpu: bool


@attr.s(auto_attribs=True, frozen=True)
class TFLitePackage(TFLitePackageInfo, Package):
    model_file: Path

    @property
    def files(self) -> List[Path]:
        return [self.model_file]
