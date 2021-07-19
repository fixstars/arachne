from typing import List, Tuple, Type

from .stage import Parameter, Stage

Pipeline = List[Tuple[Type[Stage], Parameter]]
