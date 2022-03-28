import os
import platform
import subprocess


def get_tensorrt_version() -> str:
    """
    The function for retrieving the TensorRT version

    Returns:
        str: the TensorRT version
    """
    dist = platform.linux_distribution()[0]
    if dist == "Ubuntu" or dist == "Debian":
        result = subprocess.check_output(
            "dpkg -l | grep libnvinfer-dev", shell=True, env={"PATH": os.environ["PATH"]}
        )
        return result.decode().strip().split()[2]
    else:
        # TODO: Support Fedora (RedHat)
        assert False, "Unsupported OS distribution"


def get_cuda_version() -> str:
    """
    The function for returning the CUDA version

    Returns:
        str: the CUDA version
    """
    result = subprocess.check_output("nvcc --version", shell=True, env={"PATH": os.environ["PATH"]})
    return result.decode().strip().split("\n")[-1].replace(",", "").split()[-2]


def get_cudnn_version() -> str:
    """
    The function for returning the cuDNN version

    Returns:
        str: the cuDNN version
    """
    dist = platform.linux_distribution()[0]
    if dist == "Ubuntu" or dist == "Debian":
        result = subprocess.check_output(
            "dpkg -l | grep libcudnn", shell=True, env={"PATH": os.environ["PATH"]}
        )
        return result.decode().strip().split()[2]
    else:
        # TODO: Support Fedora (RedHat)
        assert False, "Unsupported OS distribution"


def get_torch2trt_version() -> str:
    """
    The function for getting the Torch2TRT version

    Returns:
        str: the Torch2TRT version
    """
    result = subprocess.check_output(
        "pip show torch2trt", shell=True, env={"PATH": os.environ["PATH"]}
    )
    return result.decode().strip().split("\n")[1].split()[1]
