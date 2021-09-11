import tempfile
from pathlib import Path
from typing import List

import attr

from arachne.runtime.indexed_ordered_dict import IndexedOrderedDict, TensorInfoDict
from arachne.runtime.module import RuntimeModule
from arachne.runtime.module.registry import (
    get_module_class,
    module_class_list,
    register_module_class,
)
from arachne.runtime.package import Package


@attr.s(auto_attribs=True, frozen=True)
class MyPackage(Package):
    @property
    def files(self) -> List[Path]:
        return []


class MyRuntimeModule(RuntimeModule):
    def __init__(self):
        pass


register_module_class(MyPackage, MyRuntimeModule)


def test_registry_runtime_module():
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_info: TensorInfoDict = IndexedOrderedDict()
        output_info: TensorInfoDict = IndexedOrderedDict()

        package = MyPackage(dir=Path(tmp_dir), input_info=input_info, output_info=output_info)
        assert type(package) in module_class_list()

        runtime_module_class = get_module_class(package)
        assert runtime_module_class == MyRuntimeModule
