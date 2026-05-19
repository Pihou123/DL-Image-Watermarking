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

VGG perceptual loss supports three modes:

```yaml
model:
  use_vgg: true
  # pixel: RGB MSE only; vgg: VGG feature MSE only; mixed: RGB MSE + vgg_loss_weight * VGG feature MSE
  vgg_loss_mode: mixed
  vgg_loss_weight: 0.03
```

The previous `use_vgg=true` experiment used VGG feature MSE as the main image loss and improved fidelity at the cost of bit accuracy. The mixed mode keeps RGB pixel MSE and adds a small VGG feature penalty:

```text
encoder_image_loss = rgb_mse + vgg_loss_weight * vgg_feature_mse
```

Recommended MCA + Gabor + mixed VGG trial:

```bash
python scripts/train.py --config configs/base.yaml --override model.use_encoder_mca=true model.use_gabor_attention=true model.use_vgg=true model.vgg_loss_mode=mixed model.vgg_loss_weight=0.03 model.residual_embedding=false model.use_yuv_loss=false model.use_decoder_stn=false model.use_adversarial_noise_training=false model.use_decoder_diff_conv=false noise.adversarial.enabled=false noise.strategy=weighted_random
```

Residual embedding can be enabled as an experimental StegaStamp-style fidelity improvement:

```yaml
model:
  residual_embedding: true
  residual_scale: 0.5
  residual_activation: tanh
  clamp_encoded: true
```

When `residual_embedding=false`, the encoder output is used directly as before. When enabled, the encoder output is treated as a residual:

```text
encoded_image = image + residual_scale * tanh(encoder_output)
```

Recommended first trial:

```bash
python scripts/train.py --config configs/base.yaml --override model.use_vgg=false model.residual_embedding=true model.residual_scale=0.5
```

YUV color-space loss can be enabled as a StegaStamp-inspired color fidelity constraint:

```yaml
model:
  use_vgg: false
  residual_embedding: false
  use_yuv_loss: true
  yuv_loss_weight: 0.2
  yuv_channel_weights: [1.0, 10.0, 10.0]
```

It keeps the original RGB pixel MSE and adds a weighted YUV loss:

```text
encoder_image_loss = rgb_mse + yuv_loss_weight * weighted_yuv_mse
```

Recommended first trial:

```bash
python scripts/train.py --config configs/base.yaml --override model.use_vgg=false model.residual_embedding=false model.use_yuv_loss=true model.yuv_loss_weight=0.2
```

Decoder differential convolution can be enabled to improve weak watermark signal extraction:

```yaml
model:
  use_decoder_diff_conv: true
  decoder_diff_layers: 3
  decoder_diff_scale: 1.0
```

Recommended first trial, isolated from previous fidelity-loss experiments:

```bash
python scripts/train.py --config configs/base.yaml --override model.use_vgg=false model.residual_embedding=false model.use_yuv_loss=false model.use_decoder_diff_conv=true model.decoder_diff_layers=3
```

STN alignment can be enabled before the decoder:

```yaml
model:
  use_decoder_stn: true
  decoder_stn_channels: 32
  decoder_stn_scale: 0.1
```

Recommended isolated trial:

```bash
python scripts/train.py --config configs/base.yaml --override model.use_decoder_stn=true model.use_encoder_mca=false model.use_gabor_attention=false model.use_decoder_diff_conv=false model.use_vgg=false model.residual_embedding=false model.use_yuv_loss=false
```

Encoder MCA multi-scale dilated convolution can be enabled:

```yaml
model:
  use_encoder_mca: true
  encoder_mca_blocks: 1
  encoder_mca_dilations: [1, 2, 5]
```

Recommended isolated trial:

```bash
python scripts/train.py --config configs/base.yaml --override model.use_encoder_mca=true model.encoder_mca_blocks=1 model.use_decoder_stn=false model.use_gabor_attention=false model.use_decoder_diff_conv=false model.use_vgg=false model.residual_embedding=false model.use_yuv_loss=false
```

Gabor attention can guide embedding toward multi-orientation texture responses:

```yaml
model:
  use_gabor_attention: true
  gabor_kernel_size: 15
  gabor_orientations: 8
  gabor_sigmas: [3.0, 5.0]
  gabor_attention_strength: 0.5
```

Recommended isolated trial:

```bash
python scripts/train.py --config configs/base.yaml --override model.use_gabor_attention=true model.use_encoder_mca=false model.use_decoder_stn=false model.use_decoder_diff_conv=false model.use_vgg=false model.residual_embedding=false model.use_yuv_loss=false
```

Curriculum noise training can be enabled by switching the noise strategy:

```bash
python scripts/train.py --config configs/base.yaml --override noise.strategy=curriculum
```

The staged schedule lives under `noise.curriculum.schedule` in `configs/base.yaml`. It supports `default_probability`, so unspecified noises can be disabled in early stages.

Adversarial noise can be enabled as an extra layer appended to the full original noise pool:

```bash
python scripts/train.py --config configs/base.yaml --override model.use_adversarial_noise_training=true noise.adversarial.enabled=true noise.adversarial.probability=0.2 noise.strategy=weighted_random model.use_decoder_stn=false model.use_encoder_mca=false model.use_gabor_attention=false model.use_decoder_diff_conv=false model.use_vgg=false model.residual_embedding=false model.use_yuv_loss=false
```

This keeps `identity/jpeg/wechat/quantization/gaussian_noise/gaussian_blur/dropout/crop/cropout/resize` and appends `adversarial` instead of replacing the noise list. The adversarial CNN uses its own optimizer to maximize message recovery loss, while the encoder/decoder learn to resist it.

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

## 9. Web UI Demonstration

Launch the Gradio interface:

```bash
python ui/app.py
```

The UI uses `configs/base.yaml` to rebuild the model structure, then loads the selected `.pth` checkpoint through `load_checkpoint()`.

Inference flow:

- Embedding: the input image is split into 64 x 64 patches, the input binary watermark is normalized to `payload_length` bits, and each patch is passed through the trained encoder. Patches are stitched back into one watermarked image.
- Extraction: the image is split into the same patch size, each patch is passed through the trained decoder, and decoded bits from all patches are merged by majority voting.
- Direct workflow: after embedding, the generated watermarked image is automatically sent to the extraction panel and the attack demonstration panel. Users do not need to save and re-upload the image for the next step.
- Bit-level evaluation: the extraction tab can optionally take the original binary watermark and report `bit_acc`, `BER`, error count, and error positions.

Attack demonstration:

- `JPEG 压缩`: save/load through JPEG with configurable quality.
- `高斯噪声`: add pixel-level Gaussian noise.
- `高斯模糊`: apply Gaussian blur.
- `缩放`: downscale and resize back.
- `中心裁剪`: crop the center region and resize back.
- `Dropout 遮挡`: randomly mask pixels with a neutral gray value.
- `颜色量化`: reduce color levels.

`微信压缩` is not simulated in the UI because the required demonstration path is the real WeChat compression workflow.
