from abc import ABCMeta, abstractclassmethod, abstractmethod, abstractstaticmethod
from pathlib import Path
from typing import Optional, Tuple

import tvm.autotvm
import tvm.driver.tvmc as tvmc
import tvm.driver.tvmc.common as tvmccommon
import tvm.driver.tvmc.frontends as tvmcfrontends

from arachne.pipeline.package import (
    DarknetPackage,
    DarknetPackageInfo,
    ONNXPackage,
    ONNXPackageInfo,
    KerasPackage,
    KerasPackageInfo,
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
from arachne.pipeline.stage.utils import (
    get_target_from_params,
    get_target_host_from_params,
)

from .._registry import register_stage, register_stage_candidate
from ..stage import Parameter, Stage


class TVMCompilerBase(Stage, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError()
    
    @staticmethod
    @abstractmethod
    def _OutputPackage(**kwargs):
        raise NotImplementedError()
    
    @staticmethod
    @abstractmethod
    def _OutputPackageInfo(**kwargs):
        raise NotImplementedError()

    @classmethod
    def get_output_info(cls, input: PackageInfo, params: Parameter) -> Optional[PackageInfo]:        
        params = TVMCompilerBase.extract_parameters(params)
        target = params["target"]
        target_host = params["target_host"]
        target_tvmdev = tvmccommon.parse_target(params["target"])[-1]["raw"]
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
                ONNXPackageInfo,
                KerasPackageInfo,
            ),
        ):
            return None
        if isinstance(input, TFLitePackageInfo) and input.for_edgetpu:
            return None
        return cls._OutputPackageInfo(target=target, target_host=target_host, target_tvmdev=target_tvmdev)

    @staticmethod
    def extract_parameters(params: Parameter) -> Parameter:
        target = get_target_from_params(params)
        target_host = get_target_host_from_params(params)

        return {"target": target, "target_host": target_host}

    @staticmethod
    def compile_model():
        raise NotImplementedError()

    @classmethod
    def process(cls, input: Package, params: Parameter, output_dir: Path) -> Package:
        params = TVMCompiler.extract_parameters(params)
        target = params["target"]
        assert target is not None
        target_host = params["target_host"]
        target_tvmdev = tvmccommon.parse_target(params["target"])[-1]["raw"]

        shape_dict = {key: tensorinfo.shape for key, tensorinfo in input.input_info.items()}
        filename = "tvm_package.tar"

        assert isinstance(
            input, (TFLitePackage, TorchScriptPackage, DarknetPackage, TF1Package, KerasPackage, ONNXPackage)
        )
        if isinstance(input, DarknetPackage):
            input_filename = input.weight_file
        elif isinstance(input, (TFLitePackage, TorchScriptPackage, TF1Package, KerasPackage, ONNXPackage)):
            input_filename = input.model_file

        if isinstance(input, TF1Package):
            # When tvmc.frontends loads a tf1 model (*.pb) that outputs multiple tensors, we have to specify output tensor names
            model = tvmcfrontends.load_model(
                str(input.dir / input_filename),
                shape_dict=shape_dict,
                outputs=input.output_info.keys(),
            )
        elif isinstance(input, ONNXPackage):
            model = tvmcfrontends.load_model(str(input.dir / input_filename), shape_dict=shape_dict, freeze_params=True)
        else:
            model = tvmcfrontends.load_model(str(input.dir / input_filename), shape_dict=shape_dict)
        
        cls.compile_model(
            model,
            target,
            target_host,
            output_dir,
            filename            
        )
        
        return cls._OutputPackage(
            dir=output_dir,
            input_info=input.input_info,
            output_info=input.output_info,
            target=target,
            target_host=target_host,
            target_tvmdev=target_tvmdev,
            package_file=Path(filename),
        )
        

class TVMCompiler(TVMCompilerBase):
    @staticmethod
    def get_name() -> str:
        return "tvm_compiler"

    @staticmethod
    def _OutputPackage(**kwargs) -> TVMPackage:
        return TVMPackage(**kwargs)
    
    @staticmethod
    def _OutputPackageInfo(**kwargs) -> TVMPackageInfo:
        return TVMPackageInfo(**kwargs)

    @staticmethod
    def compile_model(model, target: str, target_host: str, output_dir: Path, filename: str):
        tvm_target, _ = tvmccommon.target_from_cli(target)
        if tvm_target.kind.name == "cuda" and "arch" in tvm_target.attrs:
            tvm.autotvm.measure.measure_methods.set_cuda_target_arch(tvm_target.attrs["arch"])
            
        output_path = output_dir / filename
        tvmc.compiler.compile_model(
            model,
            target,
            package_path=str(output_path),
            export_format="tar",
            dump_code=["relay"],
            target_host=target_host,
            desired_layout=None,
        )

register_stage(TVMCompiler)

register_stage_candidate(TVMCompiler)
