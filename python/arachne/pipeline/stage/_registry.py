from typing import Iterable, List, Optional, Tuple, Type

from arachne.pipeline.stage import Parameter, Stage
from arachne.registry import Registry

_stage_registry: Registry[str, Type[Stage]] = Registry()
_stage_candidates: List[Tuple[Type[Stage], Parameter]] = []


def get_stage(key: str) -> Optional[Type[Stage]]:
    return _stage_registry.get(key)


def register_stage(stage: Type[Stage]):
    return _stage_registry.register(stage.get_name(), stage)


def stage_list() -> List[str]:
    return _stage_registry.list()


def register_stage_candidate(stage: Type[Stage], params: Parameter = {}):
    _stage_candidates.append((stage, params))


def stage_candidate_list() -> Iterable[Tuple[Type[Stage], Parameter]]:
    return _stage_candidates
