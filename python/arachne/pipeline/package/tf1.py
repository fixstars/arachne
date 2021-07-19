from pathlib import Path
from typing import List

import attr

from .package import Package, PackageInfo


@attr.s(auto_attribs=True, frozen=True)
class Tf1PackageInfo(PackageInfo):
    pass


@attr.s(auto_attribs=True, frozen=True)
class Tf1Package(Tf1PackageInfo, Package):
    model_file: Path

    @property
    def files(self) -> List[Path]:
        return [self.model_file]
