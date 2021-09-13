from collections import OrderedDict
from typing import Dict, Generic, List, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class Registry(Generic[K, V]):
    def __init__(self):
        self._registries: Dict[K, V] = OrderedDict()

    def register(self, key: K, value: V, override=False):
        assert override or key not in self._registries.keys()
        self._registries[key] = value
        return value

    def get(self, key: K) -> Optional[V]:
        return self._registries.get(key)

    def list(self) -> List[K]:
        return list(self._registries.keys())
