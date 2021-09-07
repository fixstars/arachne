from typing import Iterable, List, Optional, Tuple, Type

from arachne.pipeline.stage import Parameter, Stage
from arachne.types import Registry

StageRegistry = Registry[str, Type[Stage]]

stage_candidates: List[Tuple[Type[Stage], Parameter]] = []


def get_stage(key: str) -> Optional[Type[Stage]]:
    return StageRegistry.get(key)


def register_stage(stage: Type[Stage]):
    StageRegistry.register(stage.get_name(), stage)


def stage_list() -> List[str]:
    return StageRegistry.list()


def register_stage_candidate(stage: Type[Stage], params: Parameter = {}):
    stage_candidates.append((stage, params))


def stage_candidate_list() -> Iterable[Tuple[Type[Stage], Parameter]]:
    return stage_candidates
