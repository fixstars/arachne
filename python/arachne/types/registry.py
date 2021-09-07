from collections import OrderedDict, defaultdict
from typing import Callable, Dict, Generic, List, Optional, Type, TypeVar

from typing_extensions import Protocol

K = TypeVar("K")
V = TypeVar("V")


class Registry(Generic[K, V]):
    __registries: Dict[Type, Dict[K, V]] = defaultdict(OrderedDict)

    @classmethod
    def register(cls, key: K, value: V):
        assert key not in cls.__registries.keys()
        cls.__registries[cls][key] = value

    @classmethod
    def get(cls, key: str) -> Optional[V]:
        return cls.__registries[cls].get(key)

    @classmethod
    def list(cls) -> List[K]:
        return list(cls.__registries[cls].keys())
