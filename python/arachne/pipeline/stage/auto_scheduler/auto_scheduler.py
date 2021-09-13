from abc import ABCMeta
from pathlib import Path
from typing import Dict, Optional

import tvm.autotvm
import tvm.driver.tvmc.common as tvmccommon
from tvm import auto_scheduler
from tvm.driver.tvmc import TVMCModel

from arachne.logger import Logger
from arachne.pipeline.package import TVMCModelPackage, TVMCModelPackageInfo
from arachne.pipeline.stage.compiler.tvm import TVMCompilerBase
from arachne.pipeline.stage.utils import (
    get_target_from_params,
    get_target_host_from_params,
)
from arachne.runtime.session import parse_rpc_tracker_url

from .._registry import register_stage, register_stage_candidate
from ..stage import Parameter

logger = Logger.logger()


class AutoScheduler(TVMCompilerBase, metaclass=ABCMeta):
    """
    A stage for auto scheduling using tvm.auto_schedule
    This stage generates TVMCModelPackage, containing a TVMCModel and an auto scheduler records file.
    """

    @classmethod
    def get_name(cls) -> str:
        return "auto_scheduler"

    @staticmethod
    def _OutputPackage(**kwargs):
        return TVMCModelPackage(**kwargs)

    @staticmethod
    def _OutputPackageInfo(**kwargs):
        return TVMCModelPackageInfo(**kwargs)

    @classmethod
    def extract_parameters(cls, params: Parameter) -> Parameter:
        target = get_target_from_params(params)
        target_host = get_target_host_from_params(params)

        new_params = {}
        new_params["target"] = target
        new_params["target_host"] = target_host

        # Parameters for Runner
        new_params["rpc_key"] = params.get("rpc_key")
        new_params["rpc_host"] = params.get("rpc_host")
        new_params["priority"] = params.get("priority")
        new_params["n_parallel"] = params.get("n_parallel")
        new_params["timeout"] = params.get("timeout")
        new_params["number"] = params.get("number")
        new_params["repeat"] = params.get("repeat")
        new_params["min_repeat_ms"] = params.get("min_repeat_ms")
        new_params["cooldown_interval"] = params.get("cooldown_interval")
        new_params["enable_cpu_cache_flush"] = params.get("enable_cpu_cache_flush")
        new_params["num_measure_trials"] = params.get("num_measure_trials", 1000)
        new_params["early_stopping"] = params.get("early_stopping")

        # Parameters for TaskScheduler
        new_params["strategy"] = params.get("strategy")
        new_params["load_model_file"] = params.get("load_model_file")
        new_params["load_log_file"] = params.get("load_log_file")
        new_params["alpha"] = params.get("alpha")
        new_params["beta"] = params.get("beta")
        new_params["backward_window_size"] = params.get("backward_window_size")
        new_params["verbose"] = params.get("verbose")

        # Parameters for TuningOptions
        new_params["early_stopping"] = params.get("early_stopping")
        new_params["num_measures_per_round"] = params.get("num_measures_per_round")

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
        tvm_target, _ = tvmccommon.target_from_cli(target)
        if tvm_target.kind.name == "cuda" and "arch" in tvm_target.attrs:
            tvm.autotvm.measure.measure_methods.set_cuda_target_arch(tvm_target.attrs["arch"])

        mod, params = model.mod, model.params

        # Auto schedule
        rpc_key = compile_params.get("rpc_key")
        rpc_host = compile_params.get("rpc_host")

        tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target, target_host)

        runner_args_name = [
            "priority",
            "n_parallel",
            "timeout",
            "number",
            "repeat",
            "min_repeat_ms",
            "cooldown_interval",
            "enable_cpu_cache_flush"
        ]
        runner_args = {
            key: compile_params.get(key) for key in runner_args_name if compile_params.get(key) is not None
        }

        if rpc_key and rpc_host:
            host, port = parse_rpc_tracker_url(rpc_host)
            logger.debug(f"Using RPC tracker: {host}:{port} key: {rpc_key}")
            runner = auto_scheduler.RPCRunner(
                rpc_key,
                host,
                port,
                **runner_args
            )
        else:
            measure_ctx = auto_scheduler.LocalRPCMeasureContext(
                **runner_args
            )
            runner = measure_ctx.runner

        package_filename = "tvm_package.tar"
        package_path = output_dir / package_filename
        records_name = "record.log"
        records_path = output_dir / records_name

        tuner_args_name = [
            "strategy",
            "load_model_file",
            "load_log_file",
            "alpha",
            "beta",
            "backward_window_size",
            "verbose"
        ]
        tuner_args = {
            key: compile_params.get(key) for key in tuner_args_name if compile_params.get(key) is not None
        }
        tuner_option_args_name = [
            "early_stopping",
            "num_measure_trials",
            "num_measures_per_round"
        ]
        tuner_option_args = {
            key: compile_params.get(key) for key in tuner_option_args_name if compile_params.get(key) is not None
        }

        tuner = auto_scheduler.TaskScheduler(tasks, task_weights, **tuner_args)
        tune_option = auto_scheduler.TuningOptions(
            runner=runner,
            measure_callbacks=[auto_scheduler.RecordToFile(str(records_path))],
            **tuner_option_args
        )

        tuner.tune(tune_option)
        model.save(package_path)

        return {"package_file": package_path, "records_path": records_path}


register_stage(AutoScheduler)
register_stage_candidate(AutoScheduler)
