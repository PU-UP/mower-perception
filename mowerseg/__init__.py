"""Lawn-mower perception factory (Task × Model × Backend)."""

from mowerseg.infer import InferenceEngine, InferenceResult
from mowerseg.pipeline.engine import PerceptionEngine

__all__ = ["InferenceEngine", "InferenceResult", "PerceptionEngine"]
__version__ = "0.2.0"
