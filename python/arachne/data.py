from dataclasses import dataclass
from enum import Enum
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


class ModelFormat(Enum):
    """This contains DNN model formats supported in arachne."""

    TVM = 0
    TF_PB = 1
    KERAS_H5 = 2
    TF_SAVED_MODEL = 3
    TFLITE = 4
    PYTORCH = 5
    ONNX = 6
    OPENVINO = 7
    CAFFE = 8


@dataclass
class Model:
    """This represents DNN models in arachne.

    Attributes:
        path (str): The path to model file or directory.
        spec (arachne.data.ModelSpec, optional): the tensor specification  for this model.
    """

    path: str
    format: ModelFormat
    spec: Optional[ModelSpec] = None
