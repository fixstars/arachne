from pathlib import Path
from typing import List

import attr

from arachne.runtime.package import Package, PackageInfo
from arachne.types import QType


@attr.s(auto_attribs=True, frozen=True)
class TorchScriptPackageInfo(PackageInfo):
    qtype: QType


@attr.s(auto_attribs=True, frozen=True)
class TorchScriptPackage(TorchScriptPackageInfo, Package):
    model_file: Path

    @property
    def files(self) -> List[Path]:
        return [self.model_file]
