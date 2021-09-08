import tempfile
from pathlib import Path
from typing import List, Optional

import attr

from arachne.pipeline.runner import run_pipeline
from arachne.pipeline.stage import Parameter, Stage
from arachne.pipeline.stage.registry import get_stage, register_stage
from arachne.runtime.package import Package, PackageInfo, import_package
from arachne.types.indexed_ordered_dict import IndexedOrderedDict, TensorInfoDict


@attr.s(auto_attribs=True, frozen=True)
class MyPackageInfo(PackageInfo):
    message: str


@attr.s(auto_attribs=True, frozen=True)
class MyPackage(MyPackageInfo, Package):
    @property
    def files(self) -> List[Path]:
        return []


class MyStage(Stage):
    @classmethod
    def get_name(cls) -> str:
        return "mystage"

    @classmethod
    def get_output_info(cls, input: PackageInfo, params: Parameter) -> Optional[PackageInfo]:
        params = cls.extract_parameters(params)
        message = params["message"]

        return MyPackageInfo(message=message)

    @classmethod
    def extract_parameters(cls, params: Parameter) -> Parameter:
        message = params["message"]

        return {"message": message}

    @classmethod
    def process(cls, input: Package, params: Parameter, output_dir: Path) -> Package:
        params = cls.extract_parameters(params)
        message: str = params["message"]

        return MyPackage(
            dir=output_dir,
            input_info=input.input_info,
            output_info=input.output_info,
            message=message,
        )


register_stage(MyStage)


def test_mystage():
    """Tests for stage and package extension"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_info: TensorInfoDict = IndexedOrderedDict()
        output_info: TensorInfoDict = IndexedOrderedDict()
        message = "init"

        pkg = MyPackage(
            dir=Path(tmp_dir), input_info=input_info, output_info=output_info, message=message
        )

        # Export test
        export_pkg_path = Path(tmp_dir + "/exported.tar")
        pkg.export(export_pkg_path)

        import_dir = Path(tmp_dir + "/imported")
        import_pkg = import_package(export_pkg_path, import_dir)
        assert pkg.input_info == import_pkg.input_info
        assert pkg.output_info == import_pkg.output_info
        assert pkg.message == import_pkg.message

        # Compile test
        new_message = "compiled"
        mystage_param = {"message": new_message}
        pipeline = [(get_stage("mystage"), mystage_param)]
        output_pkgs = run_pipeline(pipeline, pkg, {}, tmp_dir)
        assert len(output_pkgs) > 0

        output_pkg = output_pkgs[-1]
        assert output_pkg.input_info == import_pkg.input_info
        assert output_pkg.output_info == import_pkg.output_info
        assert output_pkg.message == new_message
