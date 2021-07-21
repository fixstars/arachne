import io
import pickle
import tarfile
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import attr

from arachne.types import TensorInfoDict
from arachne.utils import make_artifact_dir

PACKAGE_INFO_NAME = "package_info.pickle"


class PackageInfo:
    pass


@attr.s(auto_attribs=True, frozen=True)
class Package(PackageInfo, metaclass=ABCMeta):
    dir: Path
    input_info: TensorInfoDict
    output_info: TensorInfoDict
    additional_info: Dict[str, Any] = attr.ib(default={}, kw_only=True)

    @property
    @abstractmethod
    def files(self) -> List[Path]:
        pass

    def export(self, output_path: Path):
        with tarfile.open(output_path, "w:gz") as tar:
            for file in self.files:
                tar.add(self.dir / file, file)

            buf = io.BytesIO()
            pickle.dump(self, buf)
            buf.seek(0)
            tar_info = tarfile.TarInfo(PACKAGE_INFO_NAME)
            tar_info.size = buf.getbuffer().nbytes
            tar.addfile(tar_info, buf)


def import_package(input_path: Path) -> Package:
    output_dir = make_artifact_dir("imported")

    with tarfile.open(input_path, "r:gz") as tar:
        tar.extractall(output_dir)

    with open(output_dir / PACKAGE_INFO_NAME, "rb") as f:
        package: Package = pickle.load(f)

    return attr.evolve(package, dir=output_dir)
