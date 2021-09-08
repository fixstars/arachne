from pathlib import Path
from typing import List

import attr

from arachne.runtime.package import Package, PackageInfo


@attr.s(auto_attribs=True, frozen=True)
class TF1PackageInfo(PackageInfo):
    pass


@attr.s(auto_attribs=True, frozen=True)
class TF1Package(TF1PackageInfo, Package):
    model_file: Path

    @property
    def files(self) -> List[Path]:
        return [self.model_file]
