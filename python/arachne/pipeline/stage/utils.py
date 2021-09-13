import importlib
import importlib.util
from typing import Any, Callable, List, Optional, Tuple

from arachne.dataset import Dataset
from arachne.runtime.indexed_ordered_dict import IndexedOrderedDict
from arachne.runtime.qtype import QType

from .stage import Parameter

module_suffix = 0


def get_function_from_string(text: str) -> Callable:
    global module_suffix
    if ":" in text:
        file_path, func_name = text.rsplit(":", 1)
        spec = importlib.util.spec_from_file_location(f"imported_module_{module_suffix}", file_path)
        module_suffix += 1
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod_name, func_name = text.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    assert isinstance(func, Callable)

    return func


def get_first_value(params: Parameter, keys: List[str], default: Any = None):
    for key in keys:
        if key in params:
            return params[key]

    return default


def get_target_from_params(params: Parameter) -> Optional[str]:
    return get_first_value(params, ["target", "_compiler_target"])


def get_target_host_from_params(params: Parameter) -> Optional[str]:
    return get_first_value(params, ["target_host", "_compiler_target_host"])


def get_qtype_from_params(params: Parameter, default: QType = QType.INT8) -> QType:
    return QType(get_first_value(params, ["qtype", "_quantizer_qtype"], default.value))


def get_make_dataset_from_params(
    params: Parameter,
) -> Tuple[Optional[Callable[[], Dataset]], str]:
    original_str = ""
    make_dataset = get_first_value(params, ["make_dataset", "_quantizer_make_dataset"])
    if isinstance(make_dataset, str):
        original_str = make_dataset
        make_dataset = get_function_from_string(make_dataset)
    if isinstance(make_dataset, Callable):
        return make_dataset, original_str

    return None, original_str


def get_preprocess_from_params(
    params: Parameter,
) -> Tuple[Optional[Callable[[Any, IndexedOrderedDict], IndexedOrderedDict]], str]:
    original_str = ""
    preprocess = get_first_value(params, ["preprocess", "_quantizer_preprocess"])
    if isinstance(preprocess, str):
        original_str = preprocess
        preprocess = get_function_from_string(preprocess)
    if isinstance(preprocess, Callable):
        return preprocess, original_str

    return None, original_str


def parse_bool(text: str) -> bool:
    return text.lower() in ["true", "on", "yes", "t", "1"]
