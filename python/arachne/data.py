from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TensorSpec:
    """This class contains the tensor information.

    Attributes:
        name (str): tensor name.
        shape (List[int]): tensor shape.
        dtype (str): tensor data type.
    """

    name: str
    shape: List[int]
    dtype: str


@dataclass
class ModelSpec:
    """This class keeps the input and output tensor information of the model.

    Attributes:
        inputs (List[arachne.data.TensorSpec]): input tensors
        outputs (List[arachne.data.TensorSpec]): output tensors
    """

    inputs: List[TensorSpec]
    outputs: List[TensorSpec]


@dataclass
class Model:
    """This represents DNN models in arachne.

    Attributes:
        path (str): The path to model file or directory.
        spec (arachne.data.ModelSpec, optional): the tensor specification  for this model.
    """

    path: str
    spec: Optional[ModelSpec] = None
