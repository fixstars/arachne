from typing import List

from arachne.utils import get_tool_config_objects

from .openvino2tf import register_openvino2tf_config
from .openvino_mo import register_openvino_mo_config
from .tflite_converter import register_tflite_converter_config
from .tftrt import register_tftrt_config
from .torch2onnx import register_torch2onnx_config

# from .torch2trt import register_torch2trt_config
from .tvm import register_tvm_config


def register_tools_config():
    register_openvino2tf_config()
    register_openvino_mo_config()
    register_tflite_converter_config()
    register_tftrt_config()
    register_torch2onnx_config()
    # register_torch2trt_config()
    register_tvm_config()


def get_all_tools() -> List[str]:
    return list(get_tool_config_objects().keys())
