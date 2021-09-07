from typing import List, Optional, Type

from arachne.pipeline.package import Package
from arachne.types import Registry

from .module import RuntimeModule

K = Type[Package]
V = Type[RuntimeModule]
ModuleClassRegistry = Registry[K, V]


def get_module_class(package: Package) -> Optional[V]:
    return ModuleClassRegistry.get(type(package))


def register_module_class(package_class: K, module_class: V):
    ModuleClassRegistry.register(package_class, module_class)


def module_class_list() -> List[K]:
    return ModuleClassRegistry.list()
