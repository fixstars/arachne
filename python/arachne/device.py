from abc import ABCMeta, abstractmethod
from typing import AbstractSet, Dict, FrozenSet, List, Optional, Set, Tuple, TypeVar

import attr
from tvm.target import Target as TVMTarget

from arachne.pipeline.package import PackageInfo, TFLitePackageInfo, TVMPackageInfo
from arachne.types import Registry


@attr.s(auto_attribs=True, frozen=True)
class Target(metaclass=ABCMeta):
    default_qtype: str

    @abstractmethod
    def validate_package(self, package: PackageInfo) -> bool:
        pass


@attr.s(auto_attribs=True, frozen=True)
class TVMCTarget(Target):
    target: str
    target_host: Optional[str] = None

    def validate_package(self, package: PackageInfo) -> bool:
        if not isinstance(package, TVMPackageInfo):
            return False

        return package.target == self.target and package.target_host == self.target_host


@attr.s(auto_attribs=True, frozen=True)
class DPUTarget(TVMCTarget):
    default_qtype: str = attr.ib(default="fp32", init=False)


@attr.s(auto_attribs=True, frozen=True)
class TFLiteTarget(Target):
    def validate_package(self, package: PackageInfo) -> bool:
        if not isinstance(package, TFLitePackageInfo):
            return False

        return not package.for_edgetpu


@attr.s(auto_attribs=True, frozen=True)
class EdgeTpuTarget(Target):
    def validate_package(self, package: PackageInfo) -> bool:
        if not isinstance(package, TFLitePackageInfo):
            return False

        return package.for_edgetpu


class Device:
    def __init__(
        self,
        name: str,
        default_features: AbstractSet[str],
        target_list: Dict[FrozenSet[str], Target],
    ):
        self._name = name
        self._target_list = target_list
        self._default_features = default_features

        if frozenset(default_features) not in target_list:
            raise ValueError("Default features are not included in the target list.")

    def get_name(self):
        return self._name

    @property
    def default_features(self) -> Set[str]:
        return set(self._default_features)

    def get_target(self, features: AbstractSet[str]) -> Optional[Target]:
        return self._target_list.get(frozenset(features))

    def get_all_target(self) -> List[Target]:
        return list(self._target_list.values())


DeviceRegistry = Registry[Device]


def parse_device_name(device_name: str) -> Tuple[str, Set[str]]:
    key, *features = map(lambda x: x.strip(), device_name.split(","))
    return key, set(features)


def get_device(key: str) -> Optional[Device]:
    return DeviceRegistry.get(key)


def get_target(device_name: str) -> Optional[Target]:
    key, features = parse_device_name(device_name)
    device = get_device(key)
    if device is None:
        return None
    if len(features) == 0:
        features = device.default_features
    return device.get_target(features)


def device_list() -> List[str]:
    return DeviceRegistry.list()


T = TypeVar("T")


def f(*args: T) -> FrozenSet[T]:
    return frozenset(args)


target_x86 = "llvm -mtriple=x86_64-linux-gnu -mattr=+fma,+avx2"
target_upcore_plus = "llvm -mtriple=x86_64-linux-gnu"
target_arm = "llvm -keys=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon"
target_raspi4 = f"{target_arm} -model=bcm2711 -mcpu=cortex-a72"
target_trt = "tensorrt --remove_no_mac_subgraphs"
target_jetson_nano = str(TVMTarget("nvidia/jetson-nano"))
target_jetson_tx2 = str(TVMTarget("nvidia/jetson-tx2"))
target_jetson_xavier_nx = "cuda -keys=cuda,gpu -arch=sm_72 -max_num_threads=1024 -max_threads_per_block=1024 -registers_per_block=65536 -shared_memory_per_block=49152 -thread_warp_size=32"

DeviceRegistry.register(
    Device(
        "host",
        {"cpu"},
        {
            f("trt", "cuda"): TVMCTarget("fp32", f"{target_trt}, cuda", target_x86),
            f("trt", "cpu"): TVMCTarget("fp32", f"{target_trt}, {target_x86}"),
            f("cuda"): TVMCTarget("fp32", "cuda", target_x86),
            f("cpu"): TVMCTarget("int8", target_x86),
            f("tflite"): TFLiteTarget("int8"),
        },
    )
)

DeviceRegistry.register(
    Device(
        "upcore-plus",
        {"cpu"},
        {
            f("cpu"): TVMCTarget("int8", target_upcore_plus),
            f("tflite"): TFLiteTarget("int8"),
        },
    )
)

DeviceRegistry.register(
    Device(
        "raspi4",
        {"cpu"},
        {
            f("cpu"): TVMCTarget("int8", target_raspi4),
            f("tflite"): TFLiteTarget("int8"),
        },
    )
)

DeviceRegistry.register(
    Device(
        "jetson-nano",
        {"trt", "cuda"},
        {
            f("trt", "cuda"): TVMCTarget("fp32", f"{target_trt}, {target_jetson_nano}", target_arm),
            f("trt", "cpu"): TVMCTarget("fp32", f"{target_trt}, {target_arm}"),
            f("cuda"): TVMCTarget("fp32", target_jetson_nano, target_arm),
            f("cpu"): TVMCTarget("int8", target_arm),
            f("tflite"): TFLiteTarget("int8"),
        },
    )
)

DeviceRegistry.register(
    Device(
        "jetson-tx2",
        {"trt", "cuda"},
        {
            f("trt", "cuda"): TVMCTarget("fp32", f"{target_trt}, {target_jetson_tx2}", target_arm),
            f("trt", "cpu"): TVMCTarget("fp32", f"{target_trt}, {target_arm}"),
            f("cuda"): TVMCTarget("fp32", target_jetson_tx2, target_arm),
            f("cpu"): TVMCTarget("int8", target_arm),
            f("tflite"): TFLiteTarget("int8"),
        },
    )
)

DeviceRegistry.register(
    Device(
        "jetson-xavier-nx",
        {"trt", "cuda"},
        {
            f("trt", "cuda"): TVMCTarget(
                "fp32", f"{target_trt}, {target_jetson_xavier_nx}", target_arm
            ),
            f("trt", "cpu"): TVMCTarget("fp32", f"{target_trt}, {target_arm}"),
            f("cuda"): TVMCTarget("fp32", target_jetson_xavier_nx, target_arm),
            f("cpu"): TVMCTarget("int8", target_arm),
            f("tflite"): TFLiteTarget("int8"),
        },
    )
)

DeviceRegistry.register(
    Device(
        "coral",
        {"edgetpu"},
        {
            f("edgetpu"): EdgeTpuTarget("int8"),
            f("cpu"): TVMCTarget("int8", target_arm),
            f("tflite"): TFLiteTarget("int8"),
        },
    )
)

DeviceRegistry.register(
    Device(
        "ultra96",
        {"dpu"},
        {
            f("dpu"): DPUTarget(f"vitis-ai -dpu=DPUCZDX8G-ultra96, {target_arm}"),
            f("cpu"): TVMCTarget("int8", target_arm),
            f("tflite"): TFLiteTarget("int8"),
        },
    )
)

DeviceRegistry.register(
    Device(
        "kv260",
        {"dpu"},
        {
            f("dpu"): DPUTarget(f"vitis-ai -dpu=DPUCZDX8G-som, {target_arm}"),
            f("cpu"): TVMCTarget("int8", target_arm),
            f("tflite"): TFLiteTarget("int8"),
        },
    )
)
