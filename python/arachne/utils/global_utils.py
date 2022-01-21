from typing import Any, Callable, Dict

_TOOL_CONFIG_GLOBAL_OBJECTS: Dict[str, Any] = {}


def get_tool_config_objects():
    return _TOOL_CONFIG_GLOBAL_OBJECTS


_TOOL_RUN_GLOBAL_OBJECTS: Dict[str, Callable] = {}


def get_tool_run_objects():
    return _TOOL_RUN_GLOBAL_OBJECTS
