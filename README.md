# DL Image Watermarking

## 中文说明

### 项目简介

这是一个基于深度学习的隐形数字水印系统，可将固定长度的二进制水印嵌入图片，并从含水印图片中提取水印。仓库包含模型训练、评估、逐噪声测试和 Gradio Web 界面。

主要功能：

- 二进制水印嵌入与提取
- 任意尺寸图片的分块处理与投票提取
- 标准分块和重叠融合两种嵌入方式
- JPEG、微信压缩、颜色量化、高斯噪声、高斯模糊、缩放、裁剪和遮挡模拟
- PNG 导出、二值图案预览和残差热力图
- 可配置的模型结构、噪声层和训练参数

本系统使用固定长度的二进制 `payload`。如需嵌入文本、编号或其他信息，应先将其编码为二进制数据。

数据集、模型权重和运行结果体积较大，不包含在仓库中。它们需要在本地准备，并会被 Git 忽略。

### 环境准备

建议使用 Python 3.10。请先根据本机 CPU 或 GPU 环境安装匹配版本的 PyTorch 和 TorchVision，再安装其余依赖：

```bash
pip install -r requirements.txt
```

可运行快速检查，确认模型和噪声层能够正常连接：

```bash
python scripts/smoke_test.py
```

### 准备训练数据

基础配置以 MIRFLICKR-25K 为参考数据集。请自行下载并解压图片，将图片文件放入：

```text
data/mirflickr25k/
```

默认使用单目录模式，程序会按照 `configs/base.yaml` 中的 `dataset.train_split` 自动划分训练集和验证集。支持 `.jpg`、`.jpeg`、`.png`、`.bmp` 和 `.webp` 图片。

也可以使用已划分的数据集。此时需要按照 TorchVision `ImageFolder` 的格式整理目录，并在配置文件中设置 `dataset.train_dir` 和 `dataset.val_dir`：

```text
data/train/<class_name>/
data/val/<class_name>/
```

水印任务不会使用类别标签，但 `ImageFolder` 模式仍要求保留至少一层类别目录。数据目录的补充说明见 `data/README.md`。

### 训练模型

使用基础配置开始训练：

```bash
python scripts/train.py --config configs/base.yaml
```

可以通过命令行覆盖配置项：

```bash
python scripts/train.py --config configs/base.yaml --override train.epochs=50 train.batch_size=16 noise.strategy=chain
```

从已有 checkpoint 继续训练：

```bash
python scripts/train.py --config <config_path> --resume <checkpoint_path>
```

每次训练都会在 `outputs/runs/` 中创建一个带时间戳的运行目录，形式如下：

```text
outputs/runs/<experiment_name>_<timestamp>/
```

其中包含：

```text
resolved_config.yaml    本次训练实际使用的配置
metrics.csv             每轮训练与验证指标
checkpoints/            模型权重
images/                 验证阶段生成的对比图
plots/                  指标曲线
logs/                   运行日志
```

### 启动 Web 界面

Web 界面需要加载本地 `.pth` 模型权重。可以使用训练过程中生成的 checkpoint，也可以使用自行准备的兼容权重。

在项目根目录运行：

```bash
python ui/app.py
```

基本流程：

1. 选择本地 `.pth` 模型权重。
2. 上传原始图片。
3. 输入仅包含 `0` 和 `1` 的二进制水印。
4. 选择有效水印 bit 数和嵌入方式。
5. 点击“嵌入水印”。
6. 在“水印提取”或“攻击演示”页面查看结果。

界面导出的 PNG 文件会保存到：

```text
outputs/ui_exports/
```

### 评估与导出

使用训练运行目录中的配置和 checkpoint 进行常规评估：

```bash
python scripts/evaluate.py --config outputs/runs/<run_dir>/resolved_config.yaml --run_dir outputs/runs/<run_dir>
```

评估结果会作为新的运行目录保存到 `outputs/runs/`。

对某次训练结果进行逐噪声评估：

```bash
python scripts/per_noise_eval.py --run_dir outputs/runs/<run_dir>
```

将某次训练目录复制到导出目录：

```bash
python scripts/export_results.py --run_dir outputs/runs/<run_dir>
```

导出的结果会保存到：

```text
outputs/exports/
```

### 目录结构

```text
configs/       配置文件
data/          本地数据集目录
docs/          补充说明
outputs/       本地训练、评估和导出结果
scripts/       训练、评估和辅助脚本
src/           核心代码
ui/            Gradio Web 界面
requirements.txt
```

---

## English

### Overview

This is a deep-learning-based invisible image watermarking system. It embeds a fixed-length binary watermark into an image and extracts the watermark from the encoded image. The repository includes model training, evaluation, per-noise testing, and a Gradio Web interface.

Main features:

- Binary watermark embedding and extraction
- Patch-based processing and voting for arbitrary-size images
- Standard tiling and overlap blending modes
- JPEG, WeChat compression, quantization, Gaussian noise, Gaussian blur, resize, crop, and occlusion simulation
- PNG export, binary pattern preview, and residual heatmaps
- Configurable model structures, noise layers, and training parameters

The system uses a fixed-length binary `payload`. Text, identifiers, or other information should be encoded as binary data before embedding.

Datasets, model weights, and generated outputs are not included in the repository because of their size. Prepare them locally; Git ignores these files.

### Environment Setup

Python 3.10 is recommended. Install a PyTorch and TorchVision build that matches the local CPU or GPU environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Run the smoke test to verify the model and noise-layer wiring:

```bash
python scripts/smoke_test.py
```

### Prepare Training Data

The base configuration uses MIRFLICKR-25K as the reference dataset. Download and extract the images locally, then place the image files under:

```text
data/mirflickr25k/
```

By default, the project uses flat-folder mode and automatically creates a training/validation split based on `dataset.train_split` in `configs/base.yaml`. Supported formats are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.

You can also use a pre-split dataset. Organize it in the TorchVision `ImageFolder` format and set `dataset.train_dir` and `dataset.val_dir` in the configuration file:

```text
data/train/<class_name>/
data/val/<class_name>/
```

The watermarking task does not use class labels, but `ImageFolder` mode still requires at least one class directory. See `data/README.md` for additional notes.

### Train a Model

Start training with the base configuration:

```bash
python scripts/train.py --config configs/base.yaml
```

Override configuration values from the command line:

```bash
python scripts/train.py --config configs/base.yaml --override train.epochs=50 train.batch_size=16 noise.strategy=chain
```

Resume from an existing checkpoint:

```bash
python scripts/train.py --config <config_path> --resume <checkpoint_path>
```

Each training session creates a timestamped run directory under `outputs/runs/`:

```text
outputs/runs/<experiment_name>_<timestamp>/
```

It contains:

```text
resolved_config.yaml    Effective configuration for the run
metrics.csv             Training and validation metrics by epoch
checkpoints/            Model weights
images/                 Validation comparison images
plots/                  Metric plots
logs/                   Runtime logs
```

### Launch the Web Interface

The Web interface requires a local `.pth` checkpoint. Use a checkpoint generated during training or provide a compatible checkpoint separately.

Run the following command from the project root:

```bash
python ui/app.py
```

Basic workflow:

1. Select a local `.pth` checkpoint.
2. Upload an input image.
3. Enter a binary watermark containing only `0` and `1`.
4. Select the effective payload size and embedding mode.
5. Click the embed button.
6. Inspect the result in the extraction or attack demonstration tab.

PNG files exported from the interface are saved under:

```text
outputs/ui_exports/
```

### Evaluate and Export

Run standard evaluation with the configuration and checkpoint from a training run:

```bash
python scripts/evaluate.py --config outputs/runs/<run_dir>/resolved_config.yaml --run_dir outputs/runs/<run_dir>
```

Evaluation artifacts are written to a new run directory under `outputs/runs/`.

Run per-noise evaluation for a training run:

```bash
python scripts/per_noise_eval.py --run_dir outputs/runs/<run_dir>
```

Copy a training run into the export directory:

```bash
python scripts/export_results.py --run_dir outputs/runs/<run_dir>
```

Exported files are saved under:

```text
outputs/exports/
```

### Project Layout

```text
configs/       Configuration files
data/          Local dataset directory
docs/          Additional documentation
outputs/       Local training, evaluation, and export results
scripts/       Training, evaluation, and utility scripts
src/           Core implementation
ui/            Gradio Web interface
requirements.txt
```
