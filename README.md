# DL Image Watermarking (Refactored HiDDeN Baseline)

This repository provides a refactored, modular implementation for a deep-learning invisible watermarking system based on HiDDeN.

## Highlights

- Clear workflow split: data preprocessing, model training, noise scheduling, evaluation, export
- Centralized configuration with CLI overrides
- Real-time training progress bar with key metrics (`loss`, `bit_acc`, `bit_error`)
- Pluggable noise architecture for easy extension
- Optional STN, MCA, Gabor attention, mixed VGG perceptual loss, curriculum noise, and adversarial noise experiments
- Visual outputs: triplet image grid, difference heatmap, training curves
- Gradio Web UI for watermark embedding, direct extraction, and attack demonstration

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

6. Launch the Web UI for demonstration:

```bash
python ui/app.py
```

In the UI, select a `.pth` checkpoint, embed a text watermark into an image, and the generated watermarked image will be synchronized directly to the extraction and attack demonstration panels. Supported local attacks include JPEG compression, Gaussian noise, Gaussian blur, resize, center crop, dropout occlusion, and color quantization. WeChat compression is intentionally left to the real WeChat workflow for demonstration.

More details:

- `docs/USAGE.md`
- `docs/NOISE_GUIDE.md`
