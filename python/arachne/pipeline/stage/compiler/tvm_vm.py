from pathlib import Path
from tensorflow.python.ops.gen_array_ops import pad_eager_fallback

from arachne.pipeline.package import (
    TVMVMPackage,
    TVMVMPackageInfo,
)

from .._registry import register_stage, register_stage_candidate
from ..stage import Parameter, Stage
from . import TVMCompilerBase


class TVMVirtualMachineCompiler(TVMCompilerBase):
    @staticmethod
    def get_name() -> str:
        return "tvm_vm_compiler"
    
    @staticmethod
    def _OutputPackage(**kwargs) -> TVMVMPackage:
        return TVMVMPackage(**kwargs)
    
    @staticmethod
    def _OutputPackageInfo(**kwargs) -> TVMVMPackageInfo:
        return TVMVMPackageInfo(**kwargs)

    @staticmethod
    def compile_model(model, target: str, target_host: str, output_dir: Path, filename: str):
        def compile_vm(model, target, target_host, package_path):
            def annotation(mod, params, target, target_host):
                import tvm.driver.tvmc.common as common
                import tvm.driver.tvmc.composite_target as composite_target
                from tvm.target import Target
                tvm_target, extra_targets = common.target_from_cli(target)
                tvm_target, target_host = Target.check_and_update_host_consist(
                    tvm_target, target_host)

                for codegen_from_cli in extra_targets:
                    codegen = composite_target.get_codegen_by_target(
                        codegen_from_cli["name"])
                    partition_function = codegen["pass_pipeline"]
                    mod, codegen_config = partition_function(
                        mod, params, **codegen_from_cli["opts"])
                return mod, params, tvm_target

            def build(mod, params, tvm_target, package_path):
                from tvm.relay import vm as relay_vm
                import tarfile
                import tempfile
                vm_exec = relay_vm.compile(mod, target=tvm_target)
                lib = vm_exec.mod
                with tempfile.TemporaryDirectory() as tmpdirname:
                    lib_name = 'mod.tar'
                    path_lib = tmpdirname + '/' + lib_name
                    lib.export_library(path_lib)
                    with tarfile.open(package_path, "w") as tar:
                        tar.add(path_lib, lib_name)

            mod, params = model.mod, model.params
            mod, params, tvm_target = annotation(mod, params, target, target_host)
            build(mod, params, tvm_target, package_path)
        
        output_path = output_dir / filename
        compile_vm(model, target, target_host, output_path)

register_stage(TVMVirtualMachineCompiler)

register_stage_candidate(TVMVirtualMachineCompiler)
