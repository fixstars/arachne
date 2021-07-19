from typing import List, Optional

from arachne.types import Registry


class Device(object):
    def __init__(
        self,
        name: str,
        target: str,
        tvmdev: str,
        default_dtype: str = "fp32",
        target_host: Optional[str] = None,
        is_edgetpu: bool = False,
        is_dpu: bool = False,
    ):
        self._name = name
        self.target = target
        self.tvmdev = tvmdev
        self.default_dtype = default_dtype
        self.target_host = target_host
        self.is_edgetpu = is_edgetpu
        self.is_dpu = is_dpu

    def get_name(self):
        return self._name


DeviceRegistry = Registry[Device]


def get_device(key: str) -> Optional[Device]:
    return DeviceRegistry.get(key)


def device_list() -> List[str]:
    return DeviceRegistry.list()


target_x86 = "llvm -mtriple=x86_64-linux-gnu -mattr=+fma,+avx2"
target_arm = "llvm -keys=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon"
target_gpu = "tensorrt --remove_no_mac_subgraphs"

DeviceRegistry.register(Device("host", target_x86, "cpu", "int8"))
DeviceRegistry.register(Device("host-gpu", f"{target_gpu}, {target_x86}", "cpu", "fp32"))

target_raspi4 = f"{target_arm} -model=bcm2711 -mcpu=cortex-a72"
DeviceRegistry.register(Device("raspi4", target_raspi4, "cpu", "int8"))

DeviceRegistry.register(Device("jetson-nano", f"{target_gpu}, {target_arm}", "cpu", "fp32"))
DeviceRegistry.register(Device("jetson-nano-cpu", target_arm, "cpu", "int8"))

DeviceRegistry.register(Device("jetson-tx2", f"{target_gpu}, {target_arm}", "cpu", "fp32"))
DeviceRegistry.register(Device("jetson-tx2-cpu", target_arm, "cpu", "int8"))

DeviceRegistry.register(Device("jetson-xavier-nx", f"{target_gpu}, {target_arm}", "cpu", "fp32"))
DeviceRegistry.register(Device("jetson-xavier-nx-cpu", target_arm, "cpu", "int8"))

DeviceRegistry.register(Device("coral", "", "cpu", "int8", is_edgetpu=True))
DeviceRegistry.register(Device("coral-cpu", target_arm, "cpu", "int8"))

DeviceRegistry.register(
    Device("ultra96", f"vitis-ai -dpu=DPUCZDX8G-ultra96, {target_arm}", "cpu", "int8", is_dpu=True)
)

DeviceRegistry.register(
    Device("kv260", f"vitis-ai -dpu=DPUCZDX8G-som, {target_arm}", "cpu", "int8", is_dpu=True)
)
