# Data Directory

## 中文说明

本目录用于存放本地训练数据，不提交实际图片文件。

基础配置以 MIRFLICKR-25K 为参考数据集。下载并解压后，将图片直接放入：

```text
data/mirflickr25k/
```

默认配置会自动划分训练集和验证集。也可以在 `configs/base.yaml` 中设置 `dataset.train_dir` 和 `dataset.val_dir`，改用 TorchVision `ImageFolder` 目录结构：

```text
data/train/<class_name>/
data/val/<class_name>/
```

## English

This directory stores local training data. Image files are intentionally not committed.

The base configuration uses MIRFLICKR-25K as the reference dataset. Download and extract it, then place the images directly under:

```text
data/mirflickr25k/
```

The default configuration creates a training/validation split automatically. Alternatively, set `dataset.train_dir` and `dataset.val_dir` in `configs/base.yaml` and use the TorchVision `ImageFolder` structure:

```text
data/train/<class_name>/
data/val/<class_name>/
```
