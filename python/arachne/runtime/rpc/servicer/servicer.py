from abc import ABCMeta, abstractmethod
from typing import List, Optional, Type

import grpc

from arachne.registry import Registry


class RuntimeServicerBase(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def register_servicer_to_server(server: grpc.Server):
        pass

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass


RuntimeServicerRegistry = Registry[str, Type[RuntimeServicerBase]]()


def get_runtime_servicer(key: str) -> Optional[Type[RuntimeServicerBase]]:
    return RuntimeServicerRegistry.get(key)


def register_runtime_servicer(servicer: Type[RuntimeServicerBase]):
    RuntimeServicerRegistry.register(servicer.get_name(), servicer)


def runtime_servicer_list() -> List[str]:
    return RuntimeServicerRegistry.list()
