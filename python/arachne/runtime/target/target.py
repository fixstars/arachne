from abc import ABCMeta, abstractmethod

import attr

from arachne.pipeline.package import PackageInfo


@attr.s(auto_attribs=True, frozen=True)
class Target(metaclass=ABCMeta):
    default_qtype: str

    @abstractmethod
    def validate_package(self, package: PackageInfo) -> bool:
        pass
