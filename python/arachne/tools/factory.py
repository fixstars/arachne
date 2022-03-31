from abc import ABCMeta, abstractmethod
from logging import getLogger
from typing import Callable, List

from hydra.core.config_store import ConfigStore

from arachne.data import Model

logger = getLogger(__name__)


class ToolConfigBase(metaclass=ABCMeta):
    pass


class ToolBase(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def run(input: Model, cfg: ToolConfigBase) -> Model:
        pass


class ToolConfigFactory:
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: ToolConfigBase) -> ToolConfigBase:
            if name in cls.registry:
                logger.warning("ToolConfig for %s already exists. Will replace it", name)
            cls.registry[name] = wrapped_class

            # Register the config into hydra as well
            cs = ConfigStore.instance()
            group_name = "tools"
            cs.store(
                group=group_name,
                name=name,
                package=f"tools.{name}",
                node=wrapped_class,
            )
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(cls, name: str, **kwargs) -> ToolConfigBase:
        config_class = cls.registry[name]
        return config_class(**kwargs)

    @classmethod
    def list(cls) -> List[str]:
        return list(cls.registry.keys())


class ToolFactory:
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: ToolBase) -> ToolBase:
            if name in cls.registry:
                logger.warning("Tool for %s already exists. Will replace it", name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(cls, name: str, **kwargs) -> ToolBase:
        config_class = cls.registry[name]
        return config_class(**kwargs)

    @classmethod
    def list(cls) -> List[str]:
        return list(cls.registry.keys())
