from typing import Any, Callable, Iterable, List, Set, Type

from arachne.dataset import Dataset
from arachne.device import Target, TVMCTarget
from arachne.logger import Logger
from arachne.types.indexed_ordered_dict import IndexedOrderedDict
from arachne.utils import make_artifact_dir

from . import Pipeline
from .package import Package, PackageInfo
from .stage import Parameter, Stage
from .stage.registry import stage_candidate_list

logger = Logger.logger()


def validate_pipeline(
    pipeline: Pipeline, input: PackageInfo, default_params: Parameter = {}
) -> bool:
    for stage, params in pipeline:
        params = dict(default_params, **params)
        output = stage.get_output_info(input, params)
        if output is None:
            logger.error(f"{stage.__name__} cannot process {repr(input)} with {repr(params)}.")
            return False
        else:
            input = output

    return True


def run_pipeline(
    pipeline: Pipeline,
    input: Package,
    default_params: Parameter = {},
    work_dir: str = None,
) -> List[Package]:
    if not validate_pipeline(pipeline, input, default_params):
        raise ValueError("Pipeline definition is invalid.")

    package_list: List[Package] = []
    package = input
    for stage, params in pipeline:
        params = dict(default_params, **params)
        output_dir = make_artifact_dir(stage.get_name(), work_dir)

        logger.info(f"Running {stage.get_name()} stage...")
        package = stage.process(package, params, output_dir)
        package_list.append(package)

    return package_list


def _make_pipeline_candidate(
    input: PackageInfo,
    target: Target,
    override_params: Parameter = {},
    exclude: Set[Type[Stage]] = set(),
) -> List[Pipeline]:
    candidate: List[Pipeline] = []
    for stage, params in stage_candidate_list():
        if stage in exclude:
            continue

        params = {**params, **override_params}
        output = stage.get_output_info(input, params)
        if output is None:
            continue

        base_pipeline: Pipeline = [(stage, params)]
        if target.validate_package(output):
            candidate.append(base_pipeline)

        exclude_ = exclude.copy()
        exclude_.add(stage)
        for pipeline in _make_pipeline_candidate(output, target, override_params, exclude_):
            candidate.append(base_pipeline + pipeline)

    return candidate


def make_params_for_target(target: Target) -> Parameter:
    params: Parameter = {"_quantizer_qtype": target.default_qtype}
    if isinstance(target, TVMCTarget):
        params["_compiler_target"] = target.target
        params["_compiler_target_host"] = target.target_host
        params["_compiler_cross"] = target.cross_compiler

    return params


def make_base_params(
    preprocess: Callable[[Any, IndexedOrderedDict], IndexedOrderedDict],
    make_dataset: Callable[[], Dataset]
) -> Parameter:
    return {"_quantizer_preprocess": preprocess, "_quantizer_make_dataset": make_dataset}


def make_pipeline_candidate(
    input: PackageInfo,
    targets: Iterable[Target],
    override_params: Parameter = {},
    exclude: Set[Type[Stage]] = set(),
) -> List[Pipeline]:
    candidate: List[Pipeline] = []
    for target in targets:
        params = {**make_params_for_target(target), **override_params}
        candidate += _make_pipeline_candidate(input, target, params, exclude)

    # remove duplication
    final_candidate: List[Pipeline] = []
    extracted_pipeline_list: List[Pipeline] = []
    for pipeline in candidate:
        extracted_pipeline = [
            (stage, stage.extract_parameters(params)) for stage, params in pipeline
        ]
        if extracted_pipeline in extracted_pipeline_list:
            continue
        final_candidate.append(pipeline)
        extracted_pipeline_list.append(extracted_pipeline)

    return final_candidate
