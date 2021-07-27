from pathlib import Path
from typing import Optional

import tvm.autotvm
import tvm.driver.tvmc as tvmc
import tvm.driver.tvmc.common as tvmccommon
import tvm.driver.tvmc.frontends as tvmcfrontends

from arachne.pipeline.package import (
    DarknetPackage,
    DarknetPackageInfo,
    Package,
    PackageInfo,
    TF1Package,
    TF1PackageInfo,
    TFLitePackage,
    TFLitePackageInfo,
    TorchScriptPackage,
    TorchScriptPackageInfo,
    TVMPackage,
    TVMPackageInfo,
)
from arachne.pipeline.package.keras import KerasPackage
from arachne.pipeline.stage.utils import (
    get_target_from_params,
    get_target_host_from_params,
)

from .._registry import register_stage, register_stage_candidate
from ..stage import Parameter, Stage


class TVMCompiler(Stage):
    @staticmethod
    def get_name() -> str:
        return "tvm_compiler"

    @staticmethod
    def get_output_info(input: PackageInfo, params: Parameter) -> Optional[PackageInfo]:
        params = TVMCompiler.extract_parameters(params)
        target = params["target"]
        target_host = params["target_host"]
        if target is None:
            return None
        if "vitis-ai" in target:
            return None
        if not isinstance(
            input,
            (
                TFLitePackageInfo,
                TorchScriptPackageInfo,
                DarknetPackageInfo,
                TF1PackageInfo,
                KerasPackage,
            ),
        ):
            return None
        if isinstance(input, TFLitePackageInfo) and input.for_edgetpu:
            return None

        return TVMPackageInfo(target=target, target_host=target_host)

    @staticmethod
    def extract_parameters(params: Parameter) -> Parameter:
        target = get_target_from_params(params)
        target_host = get_target_host_from_params(params)

        return {"target": target, "target_host": target_host}

    @staticmethod
    def process(input: Package, params: Parameter, output_dir: Path) -> Package:
        params = TVMCompiler.extract_parameters(params)
        target = params["target"]
        assert target is not None
        target_host = params["target_host"]

        shape_dict = {key: tensorinfo.shape for key, tensorinfo in input.input_info.items()}
        filename = "tvm_package.tar"
        output_path = output_dir / filename

        assert isinstance(
            input, (TFLitePackage, TorchScriptPackage, DarknetPackage, TF1Package, KerasPackage)
        )
        if isinstance(input, DarknetPackage):
            input_filename = input.weight_file
        elif isinstance(input, (TFLitePackage, TorchScriptPackage, TF1Package, KerasPackage)):
            input_filename = input.model_file

        if isinstance(input, TF1Package):
            # When tvmc.frontends loads a tf1 model (*.pb) that outputs multiple tensors, we have to specify output tensor names
            model = tvmcfrontends.load_model(
                str(input.dir / input_filename),
                shape_dict=shape_dict,
                outputs=input.output_info.keys(),
            )
        else:
            model = tvmcfrontends.load_model(str(input.dir / input_filename), shape_dict=shape_dict)

        tvm_target, _ = tvmccommon.target_from_cli(target)
        if tvm_target.kind.name == "cuda" and "arch" in tvm_target.attrs:
            tvm.autotvm.measure.measure_methods.set_cuda_target_arch(tvm_target.attrs["arch"])

        tvmc.compiler.compile_model(
            model,
            target,
            package_path=str(output_path),
            export_format="tar",
            dump_code=["relay"],
            target_host=target_host,
            desired_layout=None,
        )

        return TVMPackage(
            dir=output_dir,
            input_info=input.input_info,
            output_info=input.output_info,
            target=target,
            target_host=target_host,
            package_file=Path(filename),
        )


register_stage(TVMCompiler)

register_stage_candidate(TVMCompiler)
