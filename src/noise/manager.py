from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn as nn

# Register built-in layers.
from . import crop, cropout, dropout, gaussian_blur, gaussian_noise  # noqa: F401
from . import identity, jpeg_compression, quantization, resize, wechat_compress  # noqa: F401
from .registry import create_noise


class NoiseManager(nn.Module):
    """Central noise scheduler.

    Supported strategies:
      - single_random: uniform random noise per batch
      - weighted_random: weighted random by probability
      - chain: apply every configured noise in order
      - curriculum: probabilities can change by epoch schedule
    """

    def __init__(self, noise_cfg: dict[str, Any], device: torch.device):
        super().__init__()
        self.device = device
        self.noise_cfg = noise_cfg
        self.strategy = str(noise_cfg.get("strategy", "single_random")).lower()

        layer_configs = noise_cfg.get("layers", [])
        if not layer_configs:
            layer_configs = [{"name": "identity", "probability": 1.0, "params": {}}]

        self.layers = nn.ModuleDict()
        self.layer_specs: list[dict[str, Any]] = []

        for index, layer_cfg in enumerate(layer_configs):
            name = str(layer_cfg["name"]).lower()
            params = dict(layer_cfg.get("params", {}))
            module = self._build_noise_module(name=name, params=params)

            key = f"{name}_{index}"
            self.layers[key] = module
            self.layer_specs.append(
                {
                    "key": key,
                    "name": name,
                    "probability": float(layer_cfg.get("probability", 1.0)),
                }
            )

    def _build_noise_module(self, name: str, params: dict[str, Any]) -> nn.Module:
        # Try with explicit device first, then fallback for layers that do not accept it.
        with_device = dict(params)
        with_device.setdefault("device", self.device)

        try:
            return create_noise(name, **with_device)
        except TypeError as exc:
            if "device" in str(exc):
                without_device = dict(params)
                return create_noise(name, **without_device)
            raise

    def forward(self, encoded: torch.Tensor, cover: torch.Tensor, epoch: int | None = None):
        specs_to_apply = self._select_specs(epoch=epoch)
        out = encoded
        applied: list[str] = []

        for spec in specs_to_apply:
            module = self.layers[spec["key"]]
            out = module(out, cover)
            applied.append(spec["name"])

        return out, {"applied_noise": applied}

    def _select_specs(self, epoch: int | None) -> list[dict[str, Any]]:
        if self.strategy == "chain":
            return self.layer_specs

        if self.strategy == "single_random":
            return [random.choice(self.layer_specs)]

        if self.strategy == "weighted_random":
            return [self._weighted_pick(self.layer_specs)]

        if self.strategy == "curriculum":
            prob_override = self._curriculum_probabilities(epoch)
            return [self._weighted_pick(self.layer_specs, prob_override=prob_override)]

        raise ValueError(f"Unknown noise strategy: {self.strategy}")

    def _weighted_pick(self, specs: list[dict[str, Any]], prob_override: dict[str, float] | None = None):
        weights = []
        for spec in specs:
            if prob_override and spec["name"] in prob_override:
                weights.append(float(prob_override[spec["name"]]))
            else:
                weights.append(float(spec["probability"]))

        total = sum(weights)
        if total <= 0:
            weights = [1.0 for _ in weights]
            total = float(len(weights))

        normalized = [w / total for w in weights]
        idx = random.choices(range(len(specs)), weights=normalized, k=1)[0]
        return specs[idx]

    def _curriculum_probabilities(self, epoch: int | None) -> dict[str, float] | None:
        cur_cfg = self.noise_cfg.get("curriculum", {})
        if not cur_cfg.get("enabled", False):
            return None

        if epoch is None:
            return None

        for stage in cur_cfg.get("schedule", []):
            start = int(stage.get("start_epoch", 1))
            end = int(stage.get("end_epoch", start))
            if start <= epoch <= end:
                return dict(stage.get("probabilities", {}))
        return None
