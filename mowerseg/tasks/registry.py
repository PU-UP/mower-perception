from __future__ import annotations

from typing import Any, Callable

from mowerseg.core.backend import InferenceBackend
from mowerseg.core.task import PerceptionTask
from mowerseg.models.registry import ModelConfig

TaskFactory = Callable[..., PerceptionTask]

_TASKS: dict[str, TaskFactory] = {}


def register_task(name: str, factory: TaskFactory) -> None:
    _TASKS[name] = factory


def create_task(
    name: str,
    model_config: ModelConfig,
    backend: InferenceBackend,
    **kwargs: Any,
) -> PerceptionTask:
    try:
        factory = _TASKS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_TASKS)) or "(none)"
        raise KeyError(f"Unsupported task '{name}'. Known: {known}") from exc
    return factory(model_config, backend, **kwargs)


def list_tasks() -> list[str]:
    return sorted(_TASKS)


def _register_builtins() -> None:
    from mowerseg.tasks.segmentation import SemanticSegmentationTask

    register_task("semantic_segmentation", SemanticSegmentationTask)


_register_builtins()
