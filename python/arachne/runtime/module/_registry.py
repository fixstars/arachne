from typing import List, Optional, Type

from arachne.runtime.package import Package
from arachne.types import Registry

from .module import RuntimeModule

_module_class_registry: Registry[Type[Package], Type[RuntimeModule]] = Registry()


def get_module_class(package: Package) -> Optional[Type[RuntimeModule]]:
    return _module_class_registry.get(type(package))


def register_module_class(package_class: Type[Package], module_class: Type[RuntimeModule]):
    _module_class_registry.register(package_class, module_class)


def module_class_list() -> List[Type[Package]]:
    return _module_class_registry.list()
