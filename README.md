# DL Image Watermarking

## 中文说明

### 项目简介

这是一个基于深度学习的隐形数字水印系统。系统可以将固定长度的二进制水印嵌入图片，并从含水印图片中提取水印。项目提供训练、评估、逐噪声测试和 Gradio Web 演示界面。

主要功能：

- 二进制水印嵌入与提取
- 任意尺寸图片的分块处理
- 多个图片块的提取结果投票
- 标准分块与重叠融合两种嵌入方式
- JPEG、微信压缩、颜色量化、高斯噪声、高斯模糊、缩放、裁剪和遮挡模拟
- 二值图案展示、无损 PNG 导出和残差热力图
- 可配置的 Encoder、Decoder、噪声层和训练参数

本项目使用固定长度的二进制 payload。需要嵌入文字、编号或其他信息时，应先将其映射为二进制数据。

### 环境准备

建议使用 Python 3.10。请根据本机 CPU 或 GPU 环境单独安装匹配版本的 PyTorch 和 TorchVision，再安装其他依赖：

```bash
pip install -r requirements.txt
```

### 启动 Web 界面

在项目根目录运行：

```bash
python ui/app.py
```

使用步骤：

1. 选择 `.pth` 模型权重文件。
2. 上传原始图片。
3. 输入仅包含 `0` 和 `1` 的二进制水印。
4. 选择有效水印 bit 数和嵌入方式。
5. 点击“嵌入水印”。
6. 在“水印提取”或“攻击演示”页面查看结果。

模型权重示例：

```text
outputs/runs/<run_name>/checkpoints/best.pth
```

### 训练模型

使用基础配置启动训练：

```bash
python scripts/train.py --config configs/base.yaml
```

通过命令行覆盖配置：

```bash
python scripts/train.py --config configs/base.yaml --override train.epochs=50 train.batch_size=16 noise.strategy=chain
```

从 checkpoint 继续训练：

```bash
python scripts/train.py --config configs/base.yaml --resume outputs/runs/<run_name>/checkpoints/best.pth
```

### 评估模型

常规评估：

```bash
python scripts/evaluate.py --config configs/base.yaml --checkpoint outputs/runs/<run_name>/checkpoints/best.pth
```

逐噪声评估：

```bash
python scripts/per_noise_eval.py --run_dir outputs/runs/<run_name>
```

导出某次训练结果：

```bash
python scripts/export_results.py --run_dir outputs/runs/<run_name>
```

### 目录结构

```text
configs/    配置文件
docs/       补充说明
scripts/    训练、评估和辅助脚本
src/        核心代码
ui/         Gradio Web 界面
data/       本地数据集和测试图片
outputs/    训练结果和导出文件
```

更多说明：

- `docs/USAGE.md`
- `docs/NOISE_GUIDE.md`

---

## English

### Overview

This project is a deep-learning-based invisible image watermarking system. It embeds a fixed-length binary watermark into an image and extracts the watermark from the encoded image. The repository includes training, evaluation, per-noise testing, and a Gradio Web interface.

Main features:

- Binary watermark embedding and extraction
- Patch-based processing for arbitrary-size images
- Majority voting across decoded patches
- Standard tiling and overlap blending modes
- JPEG, WeChat compression, quantization, Gaussian noise, Gaussian blur, resize, crop, and occlusion simulation
- Binary pattern preview, lossless PNG export, and residual heatmaps
- Configurable encoders, decoders, noise layers, and training parameters

The system uses a fixed-length binary payload. Text, identifiers, or other information should be mapped to binary data before embedding.

### Environment Setup

Python 3.10 is recommended. Install a PyTorch and TorchVision build that matches the local CPU or GPU environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

### Launch the Web Interface

Run the following command from the project root:

```bash
python ui/app.py
```

Basic workflow:

1. Select a `.pth` checkpoint.
2. Upload an input image.
3. Enter a binary watermark containing only `0` and `1`.
4. Select the effective payload size and embedding mode.
5. Click the embed button.
6. Inspect the result in the extraction or attack demonstration tab.

Checkpoint example:

```text
outputs/runs/<run_name>/checkpoints/best.pth
```

### Train a Model

Start training with the base configuration:

```bash
python scripts/train.py --config configs/base.yaml
```

Override configuration values from the command line:

```bash
python scripts/train.py --config configs/base.yaml --override train.epochs=50 train.batch_size=16 noise.strategy=chain
```

Resume from a checkpoint:

```bash
python scripts/train.py --config configs/base.yaml --resume outputs/runs/<run_name>/checkpoints/best.pth
```

### Evaluate a Model

Run standard evaluation:

```bash
python scripts/evaluate.py --config configs/base.yaml --checkpoint outputs/runs/<run_name>/checkpoints/best.pth
```

Run per-noise evaluation:

```bash
python scripts/per_noise_eval.py --run_dir outputs/runs/<run_name>
```

Export a training run:

```bash
python scripts/export_results.py --run_dir outputs/runs/<run_name>
```

### Project Layout

```text
configs/    Configuration files
docs/       Additional documentation
scripts/    Training, evaluation, and utility scripts
src/        Core implementation
ui/         Gradio Web interface
data/       Local datasets and test images
outputs/    Training runs and exported files
```

Additional documentation:

- `docs/USAGE.md`
- `docs/NOISE_GUIDE.md`
