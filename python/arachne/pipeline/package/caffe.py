from pathlib import Path
from typing import List

import attr

from arachne.runtime.package import Package, PackageInfo


@attr.s(auto_attribs=True, frozen=True)
class CaffePackageInfo(PackageInfo):
    pass


@attr.s(auto_attribs=True, frozen=True)
class CaffePackage(CaffePackageInfo, Package):
    prototxt_file: Path
    caffemodel_file: Path

    @property
    def files(self) -> List[Path]:
        return [self.prototxt_file, self.caffemodel_file]
