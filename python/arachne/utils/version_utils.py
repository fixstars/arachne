import platform
import subprocess


def get_tensorrt_version():
    dist = platform.linux_distribution()[0]
    if dist == "Ubuntu" or dist == "Debian":
        result = subprocess.check_output("dpkg -l | grep libnvinfer-dev", shell=True)
        return result.decode().strip().split()[2]
    else:
        # TODO: Support Fedora (RedHat)
        assert False, "Unsupported OS distribution"


def get_cuda_version():
    result = subprocess.check_output("nvcc --version", shell=True)
    return result.decode().strip().split("\n")[-1].replace(",", "").split()[-2]


def get_cudnn_version():
    dist = platform.linux_distribution()[0]
    if dist == "Ubuntu" or dist == "Debian":
        result = subprocess.check_output("dpkg -l | grep libcudnn", shell=True)
        return result.decode().strip().split()[2]
    else:
        # TODO: Support Fedora (RedHat)
        assert False, "Unsupported OS distribution"


def get_torch2trt_version():
    result = subprocess.check_output("pip show torch2trt", shell=True)
    return result.decode().strip().split("\n")[1].split()[1]
