# Noise Layer Guide

## Architecture

Noise is now isolated into `src/noise/` with:

- `base.py`: shared interface
- `registry.py`: plugin registration
- `manager.py`: runtime selection strategy
- built-in layers: `identity`, `jpeg`, `quantization`

## Supported Strategies

Configured by `noise.strategy` in YAML.

- `single_random`: randomly choose one noise layer (uniform)
- `weighted_random`: choose one by configured probability
- `chain`: apply all configured noise layers in order
- `curriculum`: choose one by epoch-dependent probabilities

## Config Example

```yaml
noise:
  strategy: weighted_random
  layers:
    - name: identity
      probability: 0.2
      params: {}
    - name: jpeg
      probability: 0.4
      params:
        yuv_keep_weights: [25, 9, 9]
    - name: quantization
      probability: 0.4
      params:
        fourier_terms: 10
```

Curriculum example:

```yaml
noise:
  strategy: curriculum
  layers:
    - name: identity
      probability: 0.3
      params: {}
    - name: jpeg
      probability: 0.4
      params: {}
    - name: quantization
      probability: 0.3
      params: {}
  curriculum:
    enabled: true
    schedule:
      - start_epoch: 1
        end_epoch: 10
        probabilities:
          identity: 0.7
          jpeg: 0.2
          quantization: 0.1
      - start_epoch: 11
        end_epoch: 50
        probabilities:
          identity: 0.2
          jpeg: 0.4
          quantization: 0.4
```

## Add a New Noise Layer

1. Create file under `src/noise/`, for example `gaussian_blur.py`.
2. Inherit `BaseNoiseLayer`.
3. Register with `@register_noise("gaussian_blur")`.
4. Implement `forward(encoded, cover)` and return transformed `encoded`.
5. Import module in `src/noise/__init__.py` so registration happens at startup.
6. Add it to config `noise.layers`.

Template:

```python
from __future__ import annotations

import torch
from .base import BaseNoiseLayer
from .registry import register_noise

@register_noise("your_noise_name")
class YourNoise(BaseNoiseLayer):
    def __init__(self, strength: float = 0.1, device: torch.device | None = None):
        super().__init__()
        self.strength = strength

    def forward(self, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        # Return transformed encoded tensor
        return encoded
```

## Debug Tips

- Keep tensor shape unchanged: `[B, 3, H, W]`
- Keep tensor range close to original (usually `[-1, 1]`)
- Avoid moving tensors to a different device inside `forward`
- Test new noise quickly with `python scripts/smoke_test.py`
