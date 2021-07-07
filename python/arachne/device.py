from collections import OrderedDict
from typing import List


import tvm.rpc
from tvm._ffi.runtime_ctypes import Device as TVMDevice


def create_tvmdev(device: str, session: tvm.rpc.RPCSession) -> TVMDevice:
    # TODO(Maruoka): Support more devices
    if device == "cuda":
        tvmdev = session.cuda()
    elif device == "cl":
        tvmdev = session.cl()
    else:
        assert device == "cpu"
        tvmdev = session.cpu()
    return tvmdev


class Device(object):
    def __init__(
        self,
        name: str,
        target: str,
        target_host: str,
        tvmdev: str,
        default_dtype: str = 'fp32',
        cross_compiler: str = None,
        is_tflite: bool = False,
        is_edgetpu: bool = False,
        is_dpu: bool = False
    ):
        self.name = name
        self.target = target
        self.target_host = target_host
        self.tvmdev = tvmdev
        self.default_dype = default_dtype
        self.cross_compiler = cross_compiler
        self.is_tflite = is_tflite
        self.is_edgetpu = is_edgetpu
        self.is_dpu = is_dpu


class DeviceRegistry(object):
    T = Device
    __registries = OrderedDict()

    @classmethod
    def register(cls, value: T):
        key = value.name
        assert key not in cls.__registries.keys()
        cls.__registries[key] = value

    @classmethod
    def get(cls, key: str) -> T:
        return cls.__registries[key]

    @classmethod
    def list(cls) -> List[str]:
        return cls.__registries.keys()


def get_device(key: str) -> Device:
    return DeviceRegistry.get(key)


def device_list() -> List[str]:
    return DeviceRegistry.list()


target_x86 = "llvm -mtriple=x86_64-linux-gnu -mattr=+fma,+avx2"
target_arm = "llvm -keys=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon"
target_gpu = "tensorrt --remove_no_mac_subgraphs"
cross_arm = "aarch64-linux-gnu-gcc"

DeviceRegistry.register(Device("host", target_x86, target_x86, "cpu", "int8"))
DeviceRegistry.register(
    Device("host-gpu", f"{target_gpu}, {target_x86}", target_x86, "cpu", "fp32"))
DeviceRegistry.register(
    Device("host-tflite", "", "", "cpu", "int8", is_tflite=True))

target_raspi4 = f"{target_arm} -model=bcm2711 -mcpu=cortex-a72"
DeviceRegistry.register(
    Device("raspi4", target_raspi4, target_raspi4, "cpu", "int8", cross_arm))

DeviceRegistry.register(
    Device("jetson-nano", f"{target_gpu}, {target_arm}", target_arm, "cpu", "fp32", cross_arm))
DeviceRegistry.register(
    Device("jetson-nano-cpu", target_arm, target_arm, "cpu", "int8", cross_arm))

DeviceRegistry.register(
    Device("jetson-tx2", f"{target_gpu}, {target_arm}", target_arm, "cpu", "fp32", cross_arm))
DeviceRegistry.register(
    Device("jetson-tx2-cpu", target_arm, target_arm, "cpu", "int8", cross_arm))

DeviceRegistry.register(
    Device("jetson-xavier-nx", f"{target_gpu}, {target_arm}", target_arm, "cpu", "fp32", cross_arm))
DeviceRegistry.register(
    Device("jetson-xavier-nx-cpu", target_arm, target_arm, "cpu", "int8", cross_arm))

DeviceRegistry.register(
    Device("coral", "", "", "cpu", "int8", is_edgetpu=True))
DeviceRegistry.register(
    Device("coral-cpu", target_arm, target_arm, "cpu", "int8", cross_arm))

DeviceRegistry.register(
    Device("ultra96", 'DPUCZDX8G-ultra96', target_arm, "cpu", "int8", cross_arm, is_dpu=True))

DeviceRegistry.register(
    Device("kv260", 'DPUCZDX8G-som', target_arm, "cpu", "int8", cross_arm, is_dpu=True))
