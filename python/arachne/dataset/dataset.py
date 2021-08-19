from typing import Iterator

from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class Dataset(Protocol):
    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator:
        ...


class DatasetFromSequence:
    def __init__(self, seq):
        self._seq = seq

    def __len__(self):
        return len(self._seq)

    def __iter__(self):
        return iter(self._seq)
