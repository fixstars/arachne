from pathlib import Path
from typing import List, Optional

import attr

from arachne.runtime.package import Package, PackageInfo


@attr.s(auto_attribs=True, frozen=True)
class TVMCModelPackageInfo(PackageInfo):
    target: str
    target_host: Optional[str]
    target_tvmdev: str


@attr.s(auto_attribs=True, frozen=True)
class TVMCModelPackage(TVMCModelPackageInfo, Package):
    package_file: Path
    records_file: Optional[Path]

    @property
    def files(self) -> List[Path]:
        if self.records_file:
            return [self.package_file, self.records_file]
        else:
            return [self.package_file]
