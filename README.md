# DL Image Watermarking (Refactored HiDDeN Baseline)

This repository provides a refactored, modular implementation for a deep-learning invisible watermarking system based on HiDDeN.

## Highlights

- Clear workflow split: data preprocessing, model training, noise scheduling, evaluation, export
- Centralized configuration with CLI overrides
- Real-time training progress bar with key metrics (`loss`, `bit_acc`, `bit_error`)
- Pluggable noise architecture for easy extension
- Visual outputs: triplet image grid, difference heatmap, training curves

## Quick Start

1. Keep conda env `dlwm` torch/cuda stack unchanged.
2. Install extra dependencies:

```bash
pip install -r requirements.txt
```

3. Run training:

```bash
python scripts/train.py --config configs/base.yaml
```

4. Run evaluation:

```bash
python scripts/evaluate.py --config configs/base.yaml --run_dir outputs/runs/<your_run>
```

5. Export run artifacts:

```bash
python scripts/export_results.py --run_dir outputs/runs/<your_run>
```

More details:

- `docs/USAGE.md`
- `docs/NOISE_GUIDE.md`
