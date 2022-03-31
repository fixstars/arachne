from typing import Callable, List

from arachne.runtime.rpc.logger import Logger

from .servicer import RuntimeServicerBase

logger = Logger.logger()


class RuntimeServicerBaseFactory:
    """Registry class only contains RuntimeServicerBase"""

    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: RuntimeServicerBase) -> RuntimeServicerBase:
            if name in cls.registry:
                logger.warning("Executor %s already exists. Will replace it", name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(cls, name: str, **kwargs) -> RuntimeServicerBase:
        config_class = cls.registry[name]
        return config_class(**kwargs)

    @classmethod
    def list(cls) -> List[str]:
        return list(cls.registry.keys())
