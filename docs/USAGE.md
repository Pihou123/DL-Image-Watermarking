# Usage Guide

## 1. Environment

This project is aligned to conda environment `dlwm`.
Keep these core packages unchanged:

- `pytorch=2.5.1`
- `pytorch-cuda=12.4`
- `torchvision=0.20.1`

Install extra packages (without changing torch/cuda stack):

```bash
pip install -r requirements.txt
```

## 2. Dataset Layout

This code supports two modes:

1. `ImageFolder` mode

```text
data/
  train/
    class_a/
      img_001.png
  val/
    class_a/
      img_101.png
```

2. Flat-folder mode (auto split)

```text
data/
  mirflickr25k/
    im1.jpg
    im2.jpg
    ...
```

Use flat-folder mode by setting `dataset.source_dir` and `dataset.train_split`.

## 3. Preprocessing for Mixed Image Sizes

To keep watermark performance and stability on mixed-size images, the default pipeline is:

- Train: `Resize(shorter_side) -> RandomCrop(image_size) -> Normalize`
- Val: `Resize(shorter_side) -> CenterCrop(image_size) -> Normalize`

This avoids black-border padding from `pad_if_needed` and keeps fixed model input size.

Config entries:

- `dataset.image_size`
- `dataset.preprocess.train_resize_short_side`
- `dataset.preprocess.val_resize_short_side`
- `dataset.preprocess.interpolation`
- `dataset.preprocess.antialias`
- `dataset.preprocess.random_hflip_prob`

## 4. Configure Training

Main config file: `configs/base.yaml`

Key sections:

- `dataset`: paths, split, preprocessing, loader workers
- `model`: architecture and loss weights
- `noise`: strategy and each noise layer parameters
- `train`: epochs, lr, amp, checkpoints
- `visualization`: number of saved samples

You can override any field from CLI:

```bash
python scripts/train.py --override train.epochs=50 train.batch_size=16 noise.strategy=chain dataset.preprocess.train_resize_short_side=128
```

## 5. Start Training

```bash
python scripts/train.py --config configs/base.yaml
```

Resume from checkpoint:

```bash
python scripts/train.py --config configs/base.yaml --resume outputs/runs/your_run/checkpoints/best.pth
```

Outputs are stored in:

- `outputs/runs/<experiment_timestamp>/checkpoints`
- `outputs/runs/<experiment_timestamp>/images`
- `outputs/runs/<experiment_timestamp>/metrics.csv`
- `outputs/runs/<experiment_timestamp>/plots/training_curves.png`

## 6. Evaluate

Use explicit checkpoint:

```bash
python scripts/evaluate.py --config configs/base.yaml --checkpoint outputs/runs/your_run/checkpoints/best.pth
```

Or load latest from a run folder:

```bash
python scripts/evaluate.py --config configs/base.yaml --run_dir outputs/runs/your_run
```

## 7. Export Results

```bash
python scripts/export_results.py --run_dir outputs/runs/your_run --out_root outputs/exports
```

## 8. Quick Smoke Test

```bash
python scripts/smoke_test.py
```
