import os
import subprocess
from pathlib import Path
from typing import Optional

from arachne.pipeline.package import Package, PackageInfo, TfLitePackage, TfLitePackageInfo
from arachne.types import QType

from .._registry import register_stage, register_stage_candidate
from ..stage import Parameter, Stage


class EdgeTpuCompiler(Stage):
    @staticmethod
    def get_name() -> str:
        return "edgetpu_compiler"

    @staticmethod
    def get_output_info(input: PackageInfo, params: Parameter) -> Optional[PackageInfo]:
        if not isinstance(input, TfLitePackageInfo):
            return None
        if input.qtype not in (QType.INT8, QType.INT8_FULL):
            return None
        if input.for_edgetpu:
            return None

        return TfLitePackageInfo(qtype=input.qtype, for_edgetpu=True)

    @staticmethod
    def extract_parameters(params: Parameter) -> Parameter:
        return {}

    @staticmethod
    def process(input: Package, params: Parameter, output_dir: Path) -> Package:
        assert isinstance(input, TfLitePackage)
        input_filename = input.model_file
        name, ext = os.path.splitext(input_filename)
        output_filename = name + "_edgetpu" + ext
        commands = ["edgetpu_compiler", "-o", str(output_dir), str(input.dir / input_filename)]
        subprocess.check_call(commands)

        return TfLitePackage(
            dir=output_dir,
            input_info=input.input_info,
            output_info=input.output_info,
            qtype=input.qtype,
            for_edgetpu=True,
            model_file=Path(output_filename),
        )


register_stage(EdgeTpuCompiler)

register_stage_candidate(EdgeTpuCompiler)
