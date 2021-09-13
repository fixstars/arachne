from collections import OrderedDict
from typing import MutableMapping, TypeVar

from .tensor_info import TensorInfo

KT = TypeVar("KT")
VT = TypeVar("VT")


class IndexedOrderedDict(OrderedDict, MutableMapping[KT, VT]):
    def get_by_index(self, index: int):
        return list(self.items())[index]


TensorInfoDict = IndexedOrderedDict[str, TensorInfo]
