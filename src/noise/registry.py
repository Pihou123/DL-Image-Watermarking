from __future__ import annotations

from typing import Callable

from .base import BaseNoiseLayer

_NOISE_REGISTRY: dict[str, type[BaseNoiseLayer]] = {}


def register_noise(name: str) -> Callable[[type[BaseNoiseLayer]], type[BaseNoiseLayer]]:
    key = name.lower().strip()

    def decorator(cls: type[BaseNoiseLayer]) -> type[BaseNoiseLayer]:
        _NOISE_REGISTRY[key] = cls
        return cls

    return decorator


def create_noise(name: str, **kwargs) -> BaseNoiseLayer:
    key = name.lower().strip()
    if key not in _NOISE_REGISTRY:
        available = ", ".join(sorted(_NOISE_REGISTRY.keys()))
        raise ValueError(f"Unknown noise layer '{name}'. Available: {available}")
    cls = _NOISE_REGISTRY[key]
    return cls(**kwargs)


def list_registered_noise() -> list[str]:
    return sorted(_NOISE_REGISTRY.keys())
