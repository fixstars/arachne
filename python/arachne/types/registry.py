from collections import OrderedDict, defaultdict
from typing import Callable, Dict, Generic, List, Optional, Type, TypeVar

from typing_extensions import Protocol


class HasName(Protocol):
    get_name: Callable[[], str]


T = TypeVar("T", bound=HasName)


class Registry(Generic[T]):
    __registries: Dict[Type, Dict[str, T]] = defaultdict(OrderedDict)

    @classmethod
    def register(cls, value: T):
        key = value.get_name()
        assert key not in cls.__registries.keys()
        cls.__registries[cls][key] = value

    @classmethod
    def get(cls, key: str) -> Optional[T]:
        return cls.__registries[cls].get(key)

    @classmethod
    def list(cls) -> List[str]:
        return list(cls.__registries[cls].keys())
