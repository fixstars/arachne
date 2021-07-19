from typing import List, Set, Type

from arachne.logger import Logger
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
    pipeline: Pipeline, input: Package, default_params: Parameter = {}, work_dir: str = None,
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


def make_pipeline_candidate(
    input: PackageInfo,
    targets: List[PackageInfo],
    base_params: Parameter = {},
    exclude: Set[Type[Stage]] = set(),
) -> List[Pipeline]:
    def match_package(input: PackageInfo, targets: List[PackageInfo]) -> bool:
        for target in targets:
            if input == target:
                return True
        return False

    result: List[Pipeline] = []
    for stage, params in stage_candidate_list():
        if stage in exclude:
            continue

        params = dict(base_params, **params)
        output = stage.get_output_info(input, params)
        if output is None:
            continue

        base_pipeline: Pipeline = [(stage, params)]
        if match_package(output, targets):
            result.append(base_pipeline)

        exclude_ = exclude.copy()
        exclude_.add(stage)
        for pipeline in make_pipeline_candidate(output, targets, base_params, exclude_):
            result.append(base_pipeline + pipeline)

    return result
