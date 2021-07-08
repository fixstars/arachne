from typing import List, NamedTuple


class TensorInfo(NamedTuple):
    shape: List[int]
    dtype: str = 'float32'
