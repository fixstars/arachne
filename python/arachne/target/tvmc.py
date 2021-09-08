from typing import Optional

import attr

from arachne.pipeline.package import PackageInfo, TVMPackageInfo

from .target import Target


@attr.s(auto_attribs=True, frozen=True)
class TVMCTarget(Target):
    target: str
    target_host: Optional[str] = None
    cross_compiler: Optional[str] = None

    def validate_package(self, package: PackageInfo) -> bool:
        if not isinstance(package, TVMPackageInfo):
            return False

        # NOTE: cross_compiler does not exist in package
        # So, we exclude the parameter from validatio
        return package.target == self.target and package.target_host == self.target_host


@attr.s(auto_attribs=True, frozen=True)
class DPUTarget(TVMCTarget):
    default_qtype: str = attr.ib(default="fp32", init=False)
