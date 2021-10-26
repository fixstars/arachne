import os
import tarfile
import tempfile
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import tvm.autotvm
import tvm.driver.tvmc.common as tvmccommon
import tvm.driver.tvmc.frontends as tvmcfrontends
from tvm import auto_scheduler, autotvm, relay
from tvm.driver.tvmc import composite_target
from tvm.driver.tvmc.model import TVMCModel
from tvm.runtime.vm import Executable as VMExecutable
from tvm.target import Target

from arachne.logger import Logger
from arachne.pipeline.package import (
    CaffePackage,
    CaffePackageInfo,
    DarknetPackage,
    DarknetPackageInfo,
    KerasPackage,
    KerasPackageInfo,
    TF1Package,
    TF1PackageInfo,
    TorchScriptPackage,
    TorchScriptPackageInfo,
    TVMCModelPackage,
    TVMCModelPackageInfo,
)
from arachne.pipeline.stage.utils import (
    get_target_from_params,
    get_target_host_from_params,
)
from arachne.runtime.package import (
    ONNXPackage,
    ONNXPackageInfo,
    Package,
    PackageInfo,
    TFLitePackage,
    TFLitePackageInfo,
    TVMPackage,
    TVMPackageInfo,
    TVMVMPackage,
    TVMVMPackageInfo,
)

from .._registry import register_stage, register_stage_candidate
from ..stage import Parameter, Stage

logger = Logger.logger()


class TVMCompilerBase(Stage, metaclass=ABCMeta):
    """A base class for TVMCompiler & TVMVMCompiler"""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _OutputPackage(**kwargs):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _OutputPackageInfo(**kwargs):
        raise NotImplementedError()

    @staticmethod
    def _validate_target(target: str) -> bool:
        if target is None:
            return False
        if "vitis-ai" in target:
            return False
        return True

    @staticmethod
    def _validate_input(
        input: PackageInfo, target: str, target_host: Optional[str], target_tvmdev: str
    ) -> bool:
        if not isinstance(
            input,
            (
                TFLitePackageInfo,
                TorchScriptPackageInfo,
                DarknetPackageInfo,
                TF1PackageInfo,
                ONNXPackageInfo,
                KerasPackageInfo,
                CaffePackageInfo,
                TVMCModelPackageInfo,
            ),
        ):
            return False

        if isinstance(input, TFLitePackageInfo) and input.for_edgetpu:
            return False

        if isinstance(input, TVMCModelPackageInfo) and not (
            input.target == target
            and input.target_host == target_host
            and input.target_tvmdev == target_tvmdev
        ):
            return False

        return True

    @classmethod
    def get_output_info(cls, input: PackageInfo, params: Parameter) -> Optional[PackageInfo]:
        params = cls.extract_parameters(params)
        target = params["target"]

        if not cls._validate_target(target):
            return None

        target_host = params["target_host"]
        target_tvmdev = tvmccommon.parse_target(params["target"])[-1]["raw"]

        if not cls._validate_input(input, target, target_host, target_tvmdev):
            return None

        return cls._OutputPackageInfo(
            target=target,
            target_host=target_host,
            target_tvmdev=target_tvmdev,
        )

    @classmethod
    @abstractmethod
    def extract_parameters(cls, params: Parameter) -> Parameter:
        raise NotImplementedError()

    @staticmethod
    def __load_tvmc_model(input: Package) -> TVMCModel:
        """Load a tvm.ir.IRModule and parameters as a TVMCModel
        This function usually calls tvmc.frontends.load_model()

        Parameters
        ----------
        input : a Package to be loaded

        Returns
        -------
        TVMCModel : a pair of tvm.ir.IRModule and params (dict of str to NDArray)
        """

        shape_dict = {key: tensorinfo.shape for key, tensorinfo in input.input_info.items()}

        if isinstance(input, DarknetPackage):
            input_filename = input.weight_file
        elif isinstance(input, CaffePackage):
            input_filename = input.caffemodel_file
        elif isinstance(
            input, (TFLitePackage, TorchScriptPackage, TF1Package, KerasPackage, ONNXPackage)
        ):
            input_filename = input.model_file

        if isinstance(input, TF1Package):
            # When tvmc.frontends loads a tf1 model (*.pb) that outputs multiple tensors, we have to specify output tensor names
            model = tvmcfrontends.load_model(
                str(input.dir / input_filename),
                shape_dict=shape_dict,
                outputs=input.output_info.keys(),
            )
        elif isinstance(input, ONNXPackage):
            model = tvmcfrontends.load_model(
                str(input.dir / input_filename), shape_dict=shape_dict, freeze_params=True
            )
        elif isinstance(input, TVMCModelPackage):
            model = tvmcfrontends.TVMCModel(model_path=str(input.dir / input.package_file))
        else:
            model = tvmcfrontends.load_model(str(input.dir / input_filename), shape_dict=shape_dict)

        return model

    @classmethod
    @abstractmethod
    def compile_model(
        cls,
        model: TVMCModel,
        target: str,
        target_host: str,
        output_dir: Path,
        auto_scheduler_records_path: Optional[Path],
        compile_params: Parameter,
    ) -> Dict[str, Path]:
        raise NotImplementedError()

    @classmethod
    def process(cls, input: Package, params: Parameter, output_dir: Path) -> Package:
        params = cls.extract_parameters(params)
        target = params["target"]

        if not cls._validate_target(target):
            assert False, "target must not be {}".format(target)

        target_host = params["target_host"]
        target_tvmdev = tvmccommon.parse_target(params["target"])[-1]["raw"]

        if not cls._validate_input(input, target, target_host, target_tvmdev):
            assert False, "input must not be {}".format(type(input))

        tvmc_model: TVMCModel = cls.__load_tvmc_model(input)

        auto_scheduler_record_path = None
        if isinstance(input, TVMCModelPackage) and input.records_file:
            auto_scheduler_record_path = input.dir / input.records_file

        files = cls.compile_model(
            tvmc_model, target, target_host, output_dir, auto_scheduler_record_path, params
        )

        return cls._OutputPackage(
            dir=output_dir,
            input_info=input.input_info,
            output_info=input.output_info,
            target=target,
            target_host=target_host,
            target_tvmdev=target_tvmdev,
            **files,
        )

    @staticmethod
    def _preprocess_model(mod, target: str, target_host: str, compile_params: Parameter):
        tvm_target, _ = tvmccommon.target_from_cli(target)
        if tvm_target.kind.name == "cuda" and "arch" in tvm_target.attrs:
            tvm.autotvm.measure.measure_methods.set_cuda_target_arch(tvm_target.attrs["arch"])

        desired_layout = compile_params.get("desired_layout")

        if desired_layout:
            mod = tvmccommon.convert_graph_layout(mod, desired_layout)

        return mod

    @staticmethod
    def _partition_model(mod, params: Dict, target: str, target_host: str):
        tvm_target, extra_targets = tvmccommon.target_from_cli(target)
        tvm_target, target_host = Target.check_and_update_host_consist(tvm_target, target_host)

        config = {}

        for codegen_from_cli in extra_targets:
            codegen = composite_target.get_codegen_by_target(codegen_from_cli["name"])
            partition_function = codegen["pass_pipeline"]
            mod, codegen_config = partition_function(mod, params, **codegen_from_cli["opts"])
            if codegen["config_key"] is not None:
                config[codegen["config_key"]] = (
                    codegen_config if codegen_config else codegen_from_cli["opts"]
                )

        return mod, tvm_target, config


class TVMCompiler(TVMCompilerBase):
    """A stage for compiling a dnn model via tvm.relay.build()"""

    @classmethod
    def get_name(cls) -> str:
        return "tvm_compiler"

    @staticmethod
    def _OutputPackage(**kwargs) -> TVMPackage:
        return TVMPackage(**kwargs)

    @staticmethod
    def _OutputPackageInfo(**kwargs) -> TVMPackageInfo:
        return TVMPackageInfo(**kwargs)

    @classmethod
    def extract_parameters(cls, params: Parameter) -> Parameter:
        target = get_target_from_params(params)
        target_host = get_target_host_from_params(params)

        new_params = {}
        new_params["target"] = target
        new_params["target_host"] = target_host
        new_params["tuning_records"] = params.get("tuning_records")
        new_params["disabled_pass"] = params.get("disabled_pass")
        new_params["opt_level"] = params.get("opt_level", 3)
        new_params["lib_format"] = params.get("lib_format", "tar")
        if new_params["lib_format"] == "tar":
            # NOTE: we should ignore cross and cross_options because export_package will output a *.so with a *.tar name
            new_params["cross"] = None
            new_params["cross_options"] = None
        else:
            new_params["cross"] = params.get("cross", None)
            if not new_params["cross"]:
                # if the cross parameter is not set, check the default parameter
                new_params["cross"] = params.get("_compiler_cross", None)
            new_params["cross_options"] = params.get("cross_options", None)

        return new_params

    @classmethod
    def compile_model(
        cls,
        model: TVMCModel,
        target: str,
        target_host: str,
        output_dir: Path,
        auto_scheduler_records_path: Optional[Path],
        compile_params: Parameter,
    ) -> Dict[str, Path]:
        """Compile a TVMCModel based on tvmc.compile_model()

        Parameters
        ----------
        model : TVMModel
            a pair of tvm.ir.IRModule and mod.params to be compiled

        target: str
            The target for which to compile. Can be a plain string or a path.

        target_host: str
            The target of the host machine if host-side code needs to be generated.

        output_dir: Path
            The path to generate compiled model to.

        auto_scheduler_records_path: Optional[Path]
            The path to auto scheduler records given by previous stage.

        compile_params: Parameter
            other parameters to used in compile phase.

        Returns
        -------
        files: Dict[str, Path]
            A dictionary from a name to a Path of a generated file by the compiler.
            The names must be same as the file fields of _OutputPackage.

        """
        mod, params = model.mod, model.params

        ### Partition ###
        mod = cls._preprocess_model(mod, target, target_host, compile_params)
        # Don't apply any transformations other than `preprocess_model` when auto scheduler
        # records is given, because it causes inconsistency of compiled model between two stage.
        if auto_scheduler_records_path is None:
            mod, tvm_target, tvm_config = cls._partition_model(mod, params, target, target_host)
        else:
            tvm_target, extra_targets = tvmccommon.target_from_cli(target)
            tvm_target, target_host = Target.check_and_update_host_consist(tvm_target, target_host)
            tvm_config = {}

        ### Build ###
        tuning_records = compile_params.get("tuning_records")
        disabled_pass = compile_params.get("disabled_pass")
        opt_level = compile_params.get("opt_level", 3)

        # We give priority to records in param over package's one
        if not tuning_records and auto_scheduler_records_path:
            tuning_records = str(auto_scheduler_records_path)

        if tuning_records and os.path.exists(tuning_records):
            logger.debug("tuning records file provided: %s", tuning_records)

            use_autoscheduler = tuning_records is not None
            if use_autoscheduler:
                try:
                    auto_scheduler.load_records(tuning_records)
                except tvm._ffi.base.TVMError:
                    logger.info("Cannot load tuning records: %s", tuning_records)
                    use_autoscheduler = False

            if use_autoscheduler:
                with auto_scheduler.ApplyHistoryBest(tuning_records):
                    tvm_config["relay.backend.use_auto_scheduler"] = True
                    with tvm.transform.PassContext(
                        opt_level=opt_level, config=tvm_config, disabled_pass=disabled_pass
                    ):
                        logger.debug("building relay graph with autoscheduler")
                        graph_module = relay.build(mod, target=tvm_target, params=params)
            else:
                with autotvm.apply_history_best(tuning_records):
                    with tvm.transform.PassContext(
                        opt_level=opt_level, config=tvm_config, disabled_pass=disabled_pass
                    ):
                        logger.debug("building relay graph with tuning records")
                        graph_module = relay.build(mod, target=tvm_target, params=params)
        else:
            with tvm.transform.PassContext(
                opt_level=opt_level, config=tvm_config, disabled_pass=disabled_pass
            ):
                logger.debug("building relay graph (no tuning records provided)")
                graph_module = relay.build(mod, target=tvm_target, params=params)

        ### Export a package ###
        # Create a new tvmc model package object from the graph definition.

        filename = "tvm_package.tar"
        package_path = str(output_dir / filename)

        package_path = model.export_package(
            graph_module,
            str(package_path),
            cross=compile_params["cross"],
            cross_options=compile_params["cross_options"],
            lib_format=compile_params["lib_format"],
        )

        # Write dumps to file.
        dump_code = ["relay"]
        dumps = {}
        for source_type in dump_code:
            lib = graph_module.get_lib()
            source = str(mod) if source_type == "relay" else lib.get_source(source_type)
            dumps[source_type] = source

        if dumps:
            cls.__save_dumps(package_path, dumps)

        return {"package_file": Path(filename)}

    @staticmethod
    def __save_dumps(module_name: str, dumps: Dict[str, str], dump_root: str = "."):
        """
        Serialize dump files to the disk.

        Parameters
        ----------
        module_name : str
            File name, referring to the module that generated
            the dump contents
        dumps : dict
            The output contents to be saved into the files
        dump_root : str, optional
            Path in which dump files will be created
        """

        for dump_format in dumps:
            dump_name = module_name + "." + dump_format
            with open(Path(dump_root, dump_name), "w") as f:
                f.write(dumps[dump_format])


register_stage(TVMCompiler)
register_stage_candidate(TVMCompiler)


class TVMVMCompiler(TVMCompilerBase):
    """A stage for compiling a dnn model via tvm.relay.vm.build()"""

    @classmethod
    def get_name(cls) -> str:
        return "tvm_vm_compiler"

    @staticmethod
    def _OutputPackage(**kwargs) -> TVMVMPackage:
        return TVMVMPackage(**kwargs)

    @staticmethod
    def _OutputPackageInfo(**kwargs) -> TVMVMPackageInfo:
        return TVMVMPackageInfo(**kwargs)

    @classmethod
    def extract_parameters(cls, params: Parameter) -> Parameter:
        target = get_target_from_params(params)
        target_host = get_target_host_from_params(params)

        new_params = {}
        new_params["target"] = target
        new_params["target_host"] = target_host
        new_params["disabled_pass"] = params.get("disabled_pass")
        new_params["opt_level"] = params.get("opt_level", 3)

        return new_params

    @classmethod
    def compile_model(
        cls,
        model: TVMCModel,
        target: str,
        target_host: str,
        output_dir: Path,
        auto_scheduler_records_path: Optional[Path],
        compile_params: Parameter,
    ) -> Dict[str, Path]:
        """Compile a TVMCModel into a tvm.runtime.vm.Executable tvmc.compile_model()

        Parameters
        ----------
        model : TVMModel
            a pair of tvm.ir.IRModule and mod.params to be compiled

        target: str
            The target for which to compile. Can be a plain string or a path.

        target_host: str
            The target of the host machine if host-side code needs to be generated.

        output_dir: Path
            The path to generate compiled model to.

        auto_scheduler_records_path: Optional[Path]
            The path to auto scheduler records given by previous stage.
            This parameter is just ignored in this stage.

        compile_params: Parameter
            other parameters to used in compile phase.

        Returns
        -------
        files: Dict[str, Path]
            A dictionary from a name to a Path of a generated file by the compiler.
            The names must be same as the file fields of _OutputPackage.
        """

        mod, params = model.mod, model.params

        ### Partition ###
        mod = cls._preprocess_model(mod, target, target_host, compile_params)
        mod, tvm_target, tvm_config = cls._partition_model(mod, params, target, target_host)

        ### Build ###
        disabled_pass = compile_params.get("disabled_pass")
        opt_level = compile_params.get("opt_level", 3)
        with tvm.transform.PassContext(
            opt_level=opt_level, config=tvm_config, disabled_pass=disabled_pass
        ):
            vm_exec: VMExecutable = relay.vm.compile(mod, target=tvm_target)

        filename = "tvm_package.tar"
        output_path = output_dir / filename

        ### Export a package ###
        with tempfile.TemporaryDirectory() as tmpdirname:
            lib = vm_exec.mod
            lib_name = "mod.tar"
            path_lib = tmpdirname + "/" + lib_name
            lib.export_library(path_lib)
            with tarfile.open(output_path, "w") as tar:
                tar.add(path_lib, lib_name)

        return {"package_file": filename}


register_stage(TVMVMCompiler)
register_stage_candidate(TVMVMCompiler)
