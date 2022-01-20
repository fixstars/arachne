import torch


def get_torch_dtype_from_string(dtype_str: str) -> torch.dtype:
    dtype_str_to_torch_dtype_dict = {
        "bool": torch.bool,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
    if dtype_str not in dtype_str_to_torch_dtype_dict:
        assert False, f"Not conversion map for {dtype_str}"
    return dtype_str_to_torch_dtype_dict[dtype_str]
