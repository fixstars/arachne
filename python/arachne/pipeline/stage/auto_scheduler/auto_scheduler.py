from abc import ABCMeta
from pathlib import Path
from typing import Dict, Optional

import tvm.driver.tvmc.common as tvmccommon
from tvm import auto_scheduler
from tvm.driver.tvmc import TVMCModel

from arachne.logger import Logger
from arachne.pipeline.package import TVMCModelPackage, TVMCModelPackageInfo
from arachne.pipeline.stage.compiler.tvm import TVMCompilerBase
from arachne.pipeline.stage.utils import (
    get_target_from_params,
    get_target_host_from_params,
    parse_bool,
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

    @staticmethod
    def _to_int(n: Optional[str]) -> Optional[int]:
        if n is None:
            return None
        else:
            return int(n)

    @staticmethod
    def _to_float(f: Optional[str]) -> Optional[float]:
        if f is None:
            return None
        else:
            return float(f)

    @staticmethod
    def _to_bool(b: Optional[str]) -> Optional[bool]:
        if b is None:
            return None
        else:
            return parse_bool(b)

    @classmethod
    def extract_parameters(cls, params: Parameter) -> Parameter:
        target = get_target_from_params(params)
        target_host = get_target_host_from_params(params)

        new_params: Parameter = {}
        new_params["target"] = target
        new_params["target_host"] = target_host

        # Parameters for Runner
        new_params["rpc_key"] = params.get("rpc_key")
        new_params["rpc_host"] = params.get("rpc_host")
        new_params["priority"] = cls._to_int(params.get("priority"))
        new_params["n_parallel"] = cls._to_int(params.get("n_parallel"))
        new_params["timeout"] = cls._to_int((params.get("timeout")))
        new_params["number"] = cls._to_int(params.get("number"))
        new_params["repeat"] = cls._to_int(params.get("repeat"))
        new_params["min_repeat_ms"] = cls._to_int(params.get("min_repeat_ms"))
        new_params["cooldown_interval"] = cls._to_float(params.get("cooldown_interval"))
        new_params["enable_cpu_cache_flush"] = cls._to_bool(params.get("enable_cpu_cache_flush"))

        # Parameters for TaskScheduler
        new_params["strategy"] = params.get("strategy")
        new_params["load_model_file"] = params.get("load_model_file")
        new_params["load_log_file"] = params.get("load_log_file")
        new_params["alpha"] = cls._to_float(params.get("alpha"))
        new_params["beta"] = cls._to_float(params.get("beta"))
        new_params["backward_window_size"] = cls._to_int(params.get("backward_window_size"))

        # Parameters for TuningOptions
        new_params["num_measure_trials"] = cls._to_int(params.get("num_measure_trials", "1000"))
        new_params["early_stopping"] = cls._to_int(params.get("early_stopping"))
        new_params["num_measures_per_round"] = cls._to_int(params.get("num_measures_per_round"))
        verbose = cls._to_bool(params.get("verbose"))
        if verbose is not None:
            new_params["verbose"] = 1 if verbose else 0

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
        mod, params = model.mod, model.params

        mod = cls._preprocess_model(mod, target, target_host, compile_params)

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
            "enable_cpu_cache_flush",
        ]
        runner_args = {
            key: compile_params.get(key)
            for key in runner_args_name
            if compile_params.get(key) is not None
        }

        # LocalRPCMeasureContext object. This is del'ed after used to free RPC server object.
        measure_ctx = None
        if rpc_key and rpc_host:
            host, port = parse_rpc_tracker_url(rpc_host)
            logger.debug(f"Using RPC tracker: {host}:{port} key: {rpc_key}")
            runner = auto_scheduler.RPCRunner(rpc_key, host, port, **runner_args)
        else:
            measure_ctx = auto_scheduler.LocalRPCMeasureContext(**runner_args)
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
        ]
        tuner_args = {
            key: compile_params.get(key)
            for key in tuner_args_name
            if compile_params.get(key) is not None
        }

        if "load_log_file" in tuner_args:
            import shutil

            load_log_file = tuner_args["load_log_file"]
            assert load_log_file is not None
            shutil.copy(load_log_file, records_path)

        if "load_log_file" not in tuner_args and auto_scheduler_records_path is not None:
            tuner_args["load_log_file"] = auto_scheduler_records_path

        tuner = auto_scheduler.TaskScheduler(tasks, task_weights, **tuner_args)

        tuner_option_args_name = [
            "num_measure_trials",
            "early_stopping",
            "num_measures_per_round",
            "verbose",
        ]
        tuner_option_args = {
            key: compile_params.get(key)
            for key in tuner_option_args_name
            if compile_params.get(key) is not None
        }
        tune_option = auto_scheduler.TuningOptions(
            runner=runner,
            measure_callbacks=[auto_scheduler.RecordToFile(str(records_path))],
            **tuner_option_args,
        )

        tuner.tune(tune_option)
        model.save(str(package_path))

        # Free LocalRPCMeasureContext object
        if measure_ctx is not None:
            del measure_ctx

        return {"package_file": Path(package_filename), "records_file": Path(records_name)}

    @staticmethod
    def _validate_target(target: str) -> bool:
        if not TVMCompilerBase._validate_target(target):
            return False

        tvm_target, extra_targets = tvmccommon.target_from_cli(target)
        if len(extra_targets) > 0:
            names = [t["name"] for t in extra_targets]
            logger.error(
                f"The auto scheduler stage doesn't support targets with partitioning: {names}"
            )
            return False

        return True


register_stage(AutoScheduler)
register_stage_candidate(AutoScheduler)
