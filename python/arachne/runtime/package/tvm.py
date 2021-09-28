from pathlib import Path
from typing import List, Optional

import attr

from .package import Package, PackageInfo


@attr.s(auto_attribs=True, frozen=True)
class TVMPackageInfo(PackageInfo):
    target: str
    target_host: Optional[str]
    target_tvmdev: str


@attr.s(auto_attribs=True, frozen=True)
class TVMPackage(TVMPackageInfo, Package):
    package_file: Path

    @property
    def files(self) -> List[Path]:
        return [self.package_file]