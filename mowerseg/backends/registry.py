from __future__ import annotations

from typing import Callable

from mowerseg.core.backend import InferenceBackend

BackendFactory = Callable[[], InferenceBackend]

_BACKENDS: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory) -> None:
    _BACKENDS[name] = factory


def create_backend(name: str) -> InferenceBackend:
    try:
        factory = _BACKENDS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_BACKENDS)) or "(none)"
        raise KeyError(f"Unsupported backend '{name}'. Known: {known}") from exc
    return factory()


def list_backends() -> list[str]:
    return sorted(_BACKENDS)


def _register_builtins() -> None:
    from mowerseg.backends.horizon_backend import HorizonBackend
    from mowerseg.backends.rknn_backend import RKNNBackend
    from mowerseg.backends.torch_backend import TorchBackend

    register_backend("torch", TorchBackend)
    register_backend("horizon", HorizonBackend)
    register_backend("rknn", RKNNBackend)


_register_builtins()
