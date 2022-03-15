import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import tvm
import tvm.autotvm
import tvm.target
import tvm.target.tag
from omegaconf import OmegaConf
from tvm import relay
from tvm.driver.tvmc.common import convert_graph_layout, target_from_cli
from tvm.driver.tvmc.composite_target import get_codegen_by_target
from tvm.driver.tvmc.frontends import load_model
from tvm.driver.tvmc.model import TVMCModel
from tvm.relay.backend.executor_factory import GraphExecutorFactoryModule

from arachne.tools.factory import (
    ToolBase,
    ToolConfigBase,
    ToolConfigFactory,
    ToolFactory,
)

from ..data import Model

_FACTORY_KEY = "tvm"


@ToolConfigFactory.register(_FACTORY_KEY)
@dataclass
class TVMConfig(ToolConfigBase):
    """This is a class for configuring a build process of the tvm.

    Attributes:
        cpu_target (str): A LLVM Target (e.g., x86-64, aarch64, etc). Defaults to x86-64
        cpu_attr (List[str]): List of attributes of the target (e.g., +fma, +avx2). Defaults to `[]`
        cpu_name (:obj:`str`, optional): A chip name in the current architecture (e.g., broadwell, cortex-a76, etc)
        cuda_target_device (str): A cuda target device (e.g., cuda, nvidia/nvidia-v100, etc)
        composite_target (List[str]) : TVM composite targets. Defaults to `[cpu]`
        target (:obj:`str`, optional): tvm.Target (*this parameter will be updated by the above configs*).
        target_host (:obj:`str`, optional): The host target for tvm.Target (*this parameter will be updated by the above configs*).
        desired_layout (:obj:`str`, optional): A desired graph layout (e.g., NHWC, NCHW).
        disabled_pass (:obj:`str`, optional): The list of passes that are disabled in `tvm.relay.build`.
        opt_level (int): The optimization level. Defaults to 3.
        export_format (str): The format of `TVMCModel.export_package`. Defaults to tar.
        cross_compiler (:obj:`str`, optional): The path for a cross compiler to be used when `export_format` is specified as "so".
        cross_compiler_options (:obj:`str`, optional): The option for the cross compiler.

    """

    cpu_target: str = "x86-64"
    cpu_attr: List[str] = field(default_factory=list)
    cpu_name: Optional[str] = None
    cuda_target_device: str = "cuda"
    composite_target: List[str] = field(default_factory=lambda: ["cpu"])

    # these two configs will be updated by above configurations
    target: Optional[str] = None
    target_host: Optional[str] = None

    desired_layout: Optional[str] = None
    disabled_pass: Optional[str] = None
    opt_level: int = 3
    export_format: str = "tar"
    cross_compiler: Optional[str] = None
    cross_compiler_options: Optional[str] = None


def get_predefined_config(target: str) -> TVMConfig:
    """This is a function for retrieving a pre-defined config for some targets.

    Args:
        target (int): A target name.

    Returns:
        TVMConfig: A pre-defined config object.
    """
    config_path = str(Path(__file__).parents[1]) + "/config/tvm_target/" + target + ".yaml"
    pre_defined_conf = OmegaConf.load(config_path)
    override_args = dict()
    for k in pre_defined_conf.keys():
        if "defaults" == k:
            continue
        override_args[k] = pre_defined_conf[k]  # type: ignore
    return TVMConfig(**override_args)


def _load_as_tvmc_model(input: Model) -> TVMCModel:
    assert input.spec is not None

    input_shape_dict = {}
    for ti in input.spec.inputs:
        input_shape_dict[ti.name] = list(ti.shape)
    if input.path.endswith(".pb"):
        outputs = [out.name for out in input.spec.outputs]
        model = load_model(
            path=input.path,
            shape_dict=input_shape_dict,
            outputs=outputs,
        )
    elif input.path.endswith(".onnx"):
        model = load_model(path=input.path, shape_dict=input_shape_dict, freeze_params=True)
    elif input.path.endswith(".tar"):
        model = TVMCModel(model_path=input.path)
    else:
        model = load_model(path=input.path, shape_dict=input_shape_dict)

    return model


def _add_additional_cuda_tag():
    tags = tvm.target.tag.list_tags()
    if tags and "nvidia/jetson-xavier-nx" not in tags:
        tvm.target.tag.register_tag(
            "nvidia/jetson-xavier-nx",
            config={
                "kind": "cuda",
                "arch": "sm_72",
                "shared_memory_per_block": 49152,
                "registers_per_block": 65536,
                "max_threads_per_block": 1024,
                "thread_warp_size": 32,
            },
        )


def _get_cpu_target(cfg: TVMConfig):
    mattrs = list(cfg.cpu_attr)
    if cfg.cpu_target == "x86-64":
        base = "llvm -mtriple=x86_64-linux-gnu"
        if len(mattrs) > 0:
            base = base + " -mattr=" + ",".join(mattrs)
        if cfg.cpu_name:
            base = base + " -mcpu=" + str(cfg.cpu_name)
        return base
    elif cfg.cpu_target == "aarch64":
        base = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"
        if len(mattrs) > 0:
            base = base + " -mattr=" + ",".join(mattrs)
        if cfg.cpu_name:
            base = base + " -mcpu=" + str(cfg.cpu_name)
        return base
    else:
        # TODO support other targets
        assert False, "untested cpu-target"


def _process_composite_targets(cfg: TVMConfig):
    _add_additional_cuda_tag()
    composit_targets = list(cfg.composite_target)
    if len(composit_targets) == 0:
        return
    target = []
    target_host = None
    assert len(composit_targets) <= 2, "len(composite targets) is up to 2"

    for t in composit_targets:
        if t == "tensorrt":
            target.append("tensorrt --remove_no_mac_subgraphs")
        elif t == "cuda":
            target.append(str(tvm.target.Target(cfg.cuda_target_device)))
        elif t == "cpu":
            target.append(_get_cpu_target(cfg))
        else:
            assert False, f"unsupported composite target: {t}"

    last_target = composit_targets[-1]
    if last_target != "cpu":
        target_host = _get_cpu_target(cfg)

    cfg.target = ",".join(target)
    if target_host is not None:
        cfg.target_host = target_host


def _save_relay(graph_module, module, module_path):
    dump_code = ["relay"]
    dumps = {}
    for source_type in dump_code:
        lib = graph_module.get_lib()
        source = str(module) if source_type == "relay" else lib.get_source(source_type)
        dumps[source_type] = source

    if dumps:
        for dump_format in dumps:
            dump_path = module_path + "." + dump_format
            with open(dump_path, "w") as f:
                f.write(dumps[dump_format])


@ToolFactory.register(_FACTORY_KEY)
class TVM(ToolBase):
    """This is a runner class for executing tvm.relay.build."""

    @staticmethod
    def run(input: Model, cfg: TVMConfig) -> Model:
        """
        The run method is a static method that executes tvm.relay.build() for an input model.

        Args:
            input (Model): An input model.
            cfg (TVMConfig): A config object.
        Returns:
            Model: A compiled model.
        """
        idx = itertools.count().__next__()

        # Load as a TVMC model
        tvmc_model = _load_as_tvmc_model(input)

        _process_composite_targets(cfg)

        # Check target consistency
        tvm_target, extra_targets = target_from_cli(cfg.target)
        tvm_target, target_host = tvm.target.Target.check_and_update_host_consist(
            target=tvm_target, host=cfg.target_host
        )

        module = tvmc_model.mod
        params = tvmc_model.params

        assert isinstance(tvm_target, tvm.target.Target)
        if tvm_target.kind.name == "cuda" and "arch" in tvm_target.attrs:
            tvm.autotvm.measure.measure_methods.set_cuda_target_arch(tvm_target.attrs["arch"])

        # Convert graph layout if needed
        if cfg.desired_layout:
            module = convert_graph_layout(module, cfg.desired_layout)

        # Partitioning depends on extra targets
        tvm_config = {}
        for extra in extra_targets:
            codegen = get_codegen_by_target(extra["name"])
            partition_function = codegen["pass_pipeline"]
            module, codegen_config = partition_function(module, params, **extra["opts"])
            if codegen["config_key"] is not None:
                tvm_config[codegen["config_key"]] = (
                    codegen_config if codegen_config else extra["opts"]
                )

        with tvm.transform.PassContext(
            opt_level=cfg.opt_level, config=tvm_config, disabled_pass=cfg.disabled_pass
        ):
            graph_module = relay.build(module, target=tvm_target, params=params)

        # Export as a tvm package
        filename = f"tvm_package_{idx}.tar"
        output_path = os.getcwd() + "/" + filename

        assert isinstance(graph_module, GraphExecutorFactoryModule)

        cc = None
        cc_opts = None
        if cfg.export_format == "so":
            assert cfg.cross_compiler is not None
            cc = cfg.cross_compiler
            cc_opts = cfg.cross_compiler_options

        package_path = tvmc_model.export_package(
            graph_module,
            output_path,
            cross=cc,
            cross_options=cc_opts,
            output_format=cfg.export_format,
        )

        assert package_path is not None

        _save_relay(graph_module=graph_module, module=module, module_path=package_path)

        return Model(path=package_path, spec=input.spec)
