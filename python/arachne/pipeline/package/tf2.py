from pathlib import Path
from typing import List

import attr

from .package import Package, PackageInfo


@attr.s(auto_attribs=True, frozen=True)
class Tf2PackageInfo(PackageInfo):
    pass


@attr.s(auto_attribs=True, frozen=True)
class Tf2Package(Tf2PackageInfo, Package):
    model_dir: Path

    @property
    def files(self) -> List[Path]:
        return [self.model_dir]
