"""Core perception abstractions: Frame, Result, Task, Backend."""

from mowerseg.core.backend import InferenceBackend
from mowerseg.core.frame import Frame
from mowerseg.core.result import (
    DepthResult,
    DetectionResult,
    PerceptionResult,
    PrivacyResult,
    SemanticResult,
)
from mowerseg.core.task import PerceptionTask

__all__ = [
    "DepthResult",
    "DetectionResult",
    "Frame",
    "InferenceBackend",
    "PerceptionResult",
    "PerceptionTask",
    "PrivacyResult",
    "SemanticResult",
]
