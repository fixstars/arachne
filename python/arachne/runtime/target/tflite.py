import attr

from arachne.pipeline.package import PackageInfo, TFLitePackageInfo

from .target import Target


@attr.s(auto_attribs=True, frozen=True)
class TFLiteTarget(Target):
    def validate_package(self, package: PackageInfo) -> bool:
        if not isinstance(package, TFLitePackageInfo):
            return False

        return not package.for_edgetpu


@attr.s(auto_attribs=True, frozen=True)
class EdgeTpuTarget(Target):
    def validate_package(self, package: PackageInfo) -> bool:
        if not isinstance(package, TFLitePackageInfo):
            return False

        return package.for_edgetpu
