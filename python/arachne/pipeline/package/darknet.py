from pathlib import Path
from typing import List

import attr

from .package import Package, PackageInfo


@attr.s(auto_attribs=True, frozen=True)
class DarknetPackageInfo(PackageInfo):
    pass


@attr.s(auto_attribs=True, frozen=True)
class DarknetPackage(DarknetPackageInfo, Package):
    cfg_file: Path
    weight_file: Path

    @property
    def files(self) -> List[Path]:
        return [self.cfg_file, self.weight_file]
