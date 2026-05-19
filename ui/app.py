"""基于深度学习的隐形数字水印系统 - Gradio Web 界面.

功能:
    - 水印嵌入: 上传图片，输入水印信息，生成含水印图片
    - 攻击演示: 对含水印图片执行训练噪声层模拟攻击
    - 水印提取: 从含水印图片或攻击后的图片中提取隐藏信息

启动方式:
    python ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.common.config import load_config
from src.common.runtime import resolve_device
from src.common.seed import set_seed
from src.engine.checkpoint import load_checkpoint
from src.models.hidden_system import HiddenSystem
from src.noise.manager import NoiseManager
from src.noise.registry import create_noise

DEFAULT_PATCH_SIZE = 64
DEFAULT_CONFIG = str(PROJECT_ROOT / "configs" / "base.yaml")
UI_VERSION = "binary-payload-overlap-blend-20260518"

_model_cache: dict[str, tuple[HiddenSystem, int, torch.device]] = {}


def _normalize_bit_string(message: str, length: int = 30) -> tuple[str, bool]:
    bits = "".join(ch for ch in message if ch in "01")
    if not bits:
        raise ValueError("请输入二进制水印，只能包含 0 和 1。")
    was_truncated = len(bits) > length
    bits = bits[:length].ljust(length, "0")
    return bits, was_truncated


def _msg_to_bits(message: str, length: int = 30) -> tuple[torch.Tensor, str, bool]:
    bits, was_truncated = _normalize_bit_string(message, length)
    tensor = torch.tensor([int(ch) for ch in bits], dtype=torch.float32).unsqueeze(0)
    return tensor, bits, was_truncated


def _bits_to_bit_string(bits: np.ndarray) -> str:
    return "".join(map(str, bits.astype(int).flatten()))


def _bit_eval_report(target_bits: str, extracted_bits: str) -> str:
    n = min(len(target_bits), len(extracted_bits))
    errors = [idx + 1 for idx in range(n) if target_bits[idx] != extracted_bits[idx]]
    bit_error = len(errors) / n if n > 0 else 0.0
    bit_acc = 1.0 - bit_error
    error_text = ", ".join(map(str, errors)) if errors else "无"
    return (
        f"目标 bit：{target_bits}\n"
        f"提取 bit：{extracted_bits}\n"
        f"bit_acc：{bit_acc:.2%}\n"
        f"BER：{bit_error:.2%}\n"
        f"错误 bit 数：{len(errors)} / {n}\n"
        f"错误位置：{error_text}"
    )


def _attack_strength_update(attack_type: str):
    settings = {
        "无攻击": {
            "visible": False,
            "minimum": 0,
            "maximum": 1,
            "value": 0,
            "step": 1,
            "label": "攻击强度",
        },
        "JPEG 压缩模拟": {
            "visible": False,
            "minimum": 0,
            "maximum": 1,
            "value": 0,
            "step": 1,
            "label": "使用训练配置中的 DCT/YUV JPEG 近似参数",
        },
        "微信压缩模拟": {
            "visible": False,
            "minimum": 0,
            "maximum": 1,
            "value": 0,
            "step": 1,
            "label": "使用训练配置中的 WeChat 压缩近似参数",
        },
        "颜色量化模拟": {
            "visible": False,
            "minimum": 0,
            "maximum": 1,
            "value": 0,
            "step": 1,
            "label": "使用训练配置中的 Fourier rounding 量化近似参数",
        },
        "高斯噪声": {
            "visible": True,
            "minimum": 0.0,
            "maximum": 0.2,
            "value": 0.05,
            "step": 0.005,
            "label": "Gaussian noise std（训练默认 0.05）",
        },
        "高斯模糊": {
            "visible": True,
            "minimum": 0.1,
            "maximum": 3.0,
            "value": 1.0,
            "step": 0.1,
            "label": "Gaussian blur sigma（训练默认 1.0）",
        },
        "缩放": {
            "visible": True,
            "minimum": 0.5,
            "maximum": 1.0,
            "value": 0.75,
            "step": 0.05,
            "label": "Resize ratio（训练范围 0.5-1.0）",
        },
        "随机裁剪": {
            "visible": True,
            "minimum": 0.5,
            "maximum": 1.0,
            "value": 0.7,
            "step": 0.05,
            "label": "Crop keep ratio（训练默认 0.7）",
        },
        "Dropout 遮挡": {
            "visible": True,
            "minimum": 0.1,
            "maximum": 1.0,
            "value": 0.7,
            "step": 0.05,
            "label": "Dropout keep ratio（训练默认 0.7，使用原图补区域）",
        },
        "Cropout 保留区域": {
            "visible": True,
            "minimum": 0.1,
            "maximum": 1.0,
            "value": 0.5,
            "step": 0.05,
            "label": "Cropout keep ratio（训练默认 0.5，使用原图补区域）",
        },
    }
    return gr.update(**settings.get(attack_type, settings["无攻击"]))


def _bit_accuracy(predicted: torch.Tensor, target: torch.Tensor) -> float:
    return float((predicted.detach().round().clamp(0, 1) == target.detach()).float().mean().item())


def _to_rgb_image(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")


def _pil_to_np(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _pad_to_multiple(img: Image.Image, multiple: int) -> Image.Image:
    w, h = img.size
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple
    if new_w != w or new_h != h:
        padded = Image.new("RGB", (new_w, new_h))
        padded.paste(img, (0, 0))
        if new_w > w:
            edge = img.crop((w - 1, 0, w, h))
            stretched = edge.resize((new_w - w, h), Image.LANCZOS)
            padded.paste(stretched, (w, 0))
        if new_h > h:
            edge = padded.crop((0, h - 1, new_w, h))
            stretched = edge.resize((new_w, new_h - h), Image.LANCZOS)
            padded.paste(stretched, (0, h))
        return padded
    return img


def _image_to_patches(img: Image.Image, patch_size: int) -> tuple[list[Image.Image], int, int, int, int]:
    img = _pad_to_multiple(img, patch_size)
    orig_w, orig_h = img.size
    cols = orig_w // patch_size
    rows = orig_h // patch_size
    patches = []
    for row in range(rows):
        for col in range(cols):
            box = (col * patch_size, row * patch_size, (col + 1) * patch_size, (row + 1) * patch_size)
            patches.append(img.crop(box))
    return patches, orig_w, orig_h, cols, rows


def _patches_to_image(
    patches: list[Image.Image],
    orig_w: int,
    orig_h: int,
    cols: int,
    rows: int,
    patch_size: int,
) -> Image.Image:
    out = Image.new("RGB", (orig_w, orig_h))
    idx = 0
    for row in range(rows):
        for col in range(cols):
            out.paste(patches[idx], (col * patch_size, row * patch_size))
            idx += 1
    return out


def _sliding_window_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    positions = list(range(0, max(1, length - patch_size + 1), stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def _blend_window(patch_size: int, device: torch.device) -> torch.Tensor:
    window_1d = torch.hann_window(patch_size, periodic=False, dtype=torch.float32, device=device)
    window_2d = window_1d[:, None] * window_1d[None, :]
    # Keep border weights non-zero so image borders are still covered by a single patch.
    window_2d = torch.clamp(window_2d, min=0.08)
    return window_2d.view(1, patch_size, patch_size)


def _patch_size_from_config(cfg: dict) -> int:
    image_size = cfg.get("dataset", {}).get("image_size", [DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE])
    height, width = [int(v) for v in image_size]
    if height != width:
        raise ValueError(f"当前 UI 仅支持方形 patch，得到 image_size={image_size}")
    return height


def _model_patch_size(model: HiddenSystem) -> int:
    return int(getattr(model.encoder_decoder.encoder, "height", DEFAULT_PATCH_SIZE))


def _tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    # Convert model output from [-1, 1] to a regular 8-bit RGB image through an explicit path.
    image = tensor.detach().cpu().clamp(-1, 1)
    image = (image + 1.0) / 2.0
    image = image.clamp(0, 1)
    return transforms.ToPILImage()(image)


def _encoded_tensor_to_01(tensor: torch.Tensor) -> torch.Tensor:
    image = tensor.detach().clamp(-1, 1)
    return ((image + 1.0) / 2.0).clamp(0, 1)


@torch.no_grad()
def _encode_patch(
    model: HiddenSystem,
    patch: Image.Image,
    expanded_message: torch.Tensor,
    transform: transforms.Compose,
    device: torch.device,
) -> torch.Tensor:
    tensor = transform(patch).unsqueeze(0).to(device)
    return model.encoder_decoder.encode(tensor, expanded_message)


@torch.no_grad()
def _patch_self_check(
    model: HiddenSystem,
    encoded: torch.Tensor,
    message_bits: torch.Tensor,
    transform: transforms.Compose,
    device: torch.device,
) -> tuple[float, float]:
    direct_decoded = model.encoder_decoder.decoder(encoded)
    direct_compressed, _ = model._compress_message(direct_decoded)
    direct_acc = _bit_accuracy(direct_compressed, message_bits)

    encoded_patch = _tensor_to_pil_image(encoded[0])
    roundtrip_tensor = transform(encoded_patch).unsqueeze(0).to(device)
    roundtrip_decoded = model.encoder_decoder.decoder(roundtrip_tensor)
    roundtrip_compressed, _ = model._compress_message(roundtrip_decoded)
    roundtrip_acc = _bit_accuracy(roundtrip_compressed, message_bits)
    return direct_acc, roundtrip_acc


@torch.no_grad()
def _embed_nonoverlap_image(
    model: HiddenSystem,
    img: Image.Image,
    expanded_message: torch.Tensor,
    message_bits: torch.Tensor,
    transform: transforms.Compose,
    device: torch.device,
    patch_size: int,
) -> tuple[Image.Image, int, list[float], list[float]]:
    patches, pad_w, pad_h, cols, rows = _image_to_patches(img, patch_size)
    encoded_patches = []
    direct_patch_accs = []
    roundtrip_patch_accs = []

    for patch in patches:
        encoded = _encode_patch(model, patch, expanded_message, transform, device)
        direct_acc, roundtrip_acc = _patch_self_check(model, encoded, message_bits, transform, device)
        direct_patch_accs.append(direct_acc)
        roundtrip_patch_accs.append(roundtrip_acc)
        encoded_patches.append(_tensor_to_pil_image(encoded[0]))

    full = _patches_to_image(encoded_patches, pad_w, pad_h, cols, rows, patch_size)
    return full, cols * rows, direct_patch_accs, roundtrip_patch_accs


@torch.no_grad()
def _embed_overlap_image(
    model: HiddenSystem,
    img: Image.Image,
    expanded_message: torch.Tensor,
    message_bits: torch.Tensor,
    transform: transforms.Compose,
    device: torch.device,
    patch_size: int,
    stride: int | None = None,
    self_check_limit: int = 64,
) -> tuple[Image.Image, int, list[float], list[float]]:
    if stride is None:
        stride = max(1, patch_size // 2)
    padded = _pad_to_multiple(img, patch_size)
    pad_w, pad_h = padded.size
    xs = _sliding_window_positions(pad_w, patch_size, stride)
    ys = _sliding_window_positions(pad_h, patch_size, stride)

    accum = torch.zeros(3, pad_h, pad_w, dtype=torch.float32, device=device)
    weight_sum = torch.zeros(1, pad_h, pad_w, dtype=torch.float32, device=device)
    weight = _blend_window(patch_size, device)

    direct_patch_accs = []
    roundtrip_patch_accs = []
    patch_count = 0

    for top in ys:
        for left in xs:
            patch = padded.crop((left, top, left + patch_size, top + patch_size))
            encoded = _encode_patch(model, patch, expanded_message, transform, device)
            encoded_01 = _encoded_tensor_to_01(encoded[0])
            accum[:, top : top + patch_size, left : left + patch_size] += encoded_01 * weight
            weight_sum[:, top : top + patch_size, left : left + patch_size] += weight

            if patch_count < self_check_limit:
                direct_acc, roundtrip_acc = _patch_self_check(model, encoded, message_bits, transform, device)
                direct_patch_accs.append(direct_acc)
                roundtrip_patch_accs.append(roundtrip_acc)
            patch_count += 1

    blended = (accum / weight_sum.clamp_min(1e-6)).clamp(0, 1).cpu()
    full = transforms.ToPILImage()(blended)
    return full, patch_count, direct_patch_accs, roundtrip_patch_accs


def load_model(ckpt_path: str):
    cache_key = str(Path(ckpt_path).resolve())
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    device = resolve_device("auto")
    payload = torch.load(ckpt_path, map_location=device)
    cfg = payload.get("config") or load_config(DEFAULT_CONFIG)

    image_size = tuple(cfg["dataset"].get("image_size", [64, 64]))
    noise_manager = NoiseManager(cfg["noise"], device=device).to(device)
    model = HiddenSystem(
        model_cfg=cfg["model"],
        train_cfg=cfg["train"],
        image_size=(int(image_size[0]), int(image_size[1])),
        noise_manager=noise_manager,
        device=device,
    ).to(device)

    load_checkpoint(ckpt_path, model=model, device=device, scaler=None, strict=False)
    model.eval()

    msg_len = int(cfg["model"].get("payload_length", cfg["model"]["message_length"]))
    _model_cache[cache_key] = (model, msg_len, device)
    return model, msg_len, device


def _load_checkpoint_config(ckpt_path: Optional[str]) -> dict:
    if ckpt_path is not None and Path(ckpt_path).exists():
        payload = torch.load(ckpt_path, map_location="cpu")
        return payload.get("config") or load_config(DEFAULT_CONFIG)
    return load_config(DEFAULT_CONFIG)


def _noise_params_from_config(cfg: dict, noise_name: str) -> dict:
    for layer_cfg in cfg.get("noise", {}).get("layers", []):
        if str(layer_cfg.get("name", "")).lower() == noise_name:
            return dict(layer_cfg.get("params", {}))
    return {}


def _build_frontend_noise_layer(
    attack_type: str,
    attack_strength: float,
    cfg: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, str, str]:
    attack_map = {
        "无攻击": ("identity", {}),
        "JPEG 压缩模拟": ("jpeg", _noise_params_from_config(cfg, "jpeg")),
        "微信压缩模拟": ("wechat", _noise_params_from_config(cfg, "wechat")),
        "颜色量化模拟": ("quantization", _noise_params_from_config(cfg, "quantization")),
        "高斯噪声": ("gaussian_noise", {"std": float(attack_strength)}),
        "高斯模糊": ("gaussian_blur", {"sigma": float(attack_strength)}),
        "缩放": ("resize", {"ratio_min": float(attack_strength), "ratio_max": float(attack_strength)}),
        "随机裁剪": ("crop", {"keep_ratio": float(attack_strength)}),
        "Dropout 遮挡": ("dropout", {"keep_ratio": float(attack_strength)}),
        "Cropout 保留区域": ("cropout", {"keep_ratio": float(attack_strength)}),
    }
    if attack_type not in attack_map:
        raise ValueError(f"未知攻击类型：{attack_type}")

    noise_name, params = attack_map[attack_type]
    params = dict(params)
    params.setdefault("device", device)
    layer = create_noise(noise_name, **params).to(device)

    param_text = ", ".join(f"{key}={value}" for key, value in params.items() if key != "device")
    desc = f"{noise_name}" + (f" ({param_text})" if param_text else "")
    return layer, noise_name, desc


@torch.no_grad()
def _apply_training_noise_to_image(
    image: Image.Image,
    cover: Image.Image,
    layer: torch.nn.Module,
    device: torch.device,
    patch_size: int,
) -> Image.Image:
    orig_w, orig_h = image.size
    patches, pad_w, pad_h, cols, rows = _image_to_patches(image, patch_size)
    cover = cover.resize((orig_w, orig_h), Image.BICUBIC)
    cover_patches, _, _, _, _ = _image_to_patches(cover, patch_size)

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    noised_patches = []
    for patch, cover_patch in zip(patches, cover_patches):
        encoded_tensor = tf(patch).unsqueeze(0).to(device)
        cover_tensor = tf(cover_patch).unsqueeze(0).to(device)
        noised = layer(encoded_tensor, cover_tensor)
        noised_patches.append(_tensor_to_pil_image(noised[0]))

    full = _patches_to_image(noised_patches, pad_w, pad_h, cols, rows, patch_size)
    return full.crop((0, 0, orig_w, orig_h))


@torch.no_grad()
def embed_watermark(
    image: np.ndarray,
    message: str,
    ckpt_path: Optional[str],
    inference_mode: str = "重叠融合（减轻方格）",
) -> tuple[
    Optional[np.ndarray],
    str,
    Optional[np.ndarray],
    Optional[np.ndarray],
    str,
    str,
    Optional[np.ndarray],
]:
    if image is None:
        return None, "错误：请先上传图片。", None, None, "", "", None
    if not message.strip():
        return None, "错误：请输入要嵌入的水印信息。", None, None, "", "", None
    if ckpt_path is None or not Path(ckpt_path).exists():
        return None, "错误：请先选择模型权重文件。", None, None, "", "", None

    try:
        set_seed(42)
        model, msg_len, device = load_model(ckpt_path)
        patch_size = _model_patch_size(model)
        message_bits, normalized_bits, was_truncated = _msg_to_bits(message, msg_len)
        message_bits = message_bits.to(device)

        img = _to_rgb_image(image)
        orig_w, orig_h = img.size

        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        expanded = model._expand_message(message_bits)
        if inference_mode == "标准分块（准确率优先）":
            full, patch_count, direct_patch_accs, roundtrip_patch_accs = _embed_nonoverlap_image(
                model=model,
                img=img,
                expanded_message=expanded,
                message_bits=message_bits,
                transform=tf,
                device=device,
                patch_size=patch_size,
            )
            mode_desc = f"标准 {patch_size} x {patch_size} 非重叠分块"
            self_check_desc = "全部分块"
        else:
            full, patch_count, direct_patch_accs, roundtrip_patch_accs = _embed_overlap_image(
                model=model,
                img=img,
                expanded_message=expanded,
                message_bits=message_bits,
                transform=tf,
                device=device,
                patch_size=patch_size,
            )
            mode_desc = f"重叠滑窗融合，窗口 {patch_size} x {patch_size}，stride={patch_size // 2}"
            self_check_desc = f"前 {len(direct_patch_accs)} 个滑窗分块抽样"
        full = full.crop((0, 0, orig_w, orig_h))

        result = _pil_to_np(full)
        truncation_note = "\n输入超过 payload 长度，已自动截断。" if was_truncated else ""
        info = (
            "嵌入成功！\n"
            f"UI版本：{UI_VERSION}\n"
            f"二进制水印：{normalized_bits}{truncation_note}\n"
            f"图片尺寸：{orig_w} x {orig_h}\n"
            f"推理模式：{mode_desc}\n"
            f"处理分块数：{patch_count}\n"
            f"嵌入自检范围：{self_check_desc}\n"
            f"嵌入自检 tensor 直连 bit_acc：最低 {min(direct_patch_accs):.2%}，平均 {np.mean(direct_patch_accs):.2%}\n"
            f"嵌入自检 PIL 回读 bit_acc：最低 {min(roundtrip_patch_accs):.2%}，平均 {np.mean(roundtrip_patch_accs):.2%}\n"
            "结果已自动同步到“水印提取”和“攻击演示”。"
        )
        return result, info, result, result, normalized_bits, normalized_bits, _pil_to_np(img)

    except Exception as e:
        return None, f"嵌入过程出错：{str(e)}", None, None, "", "", None


@torch.no_grad()
def extract_watermark(
    image: np.ndarray,
    ckpt_path: Optional[str],
    expected_bits: str = "",
) -> tuple[str, str, str]:
    if image is None:
        return "", "错误：请先上传、同步或生成待提取图片。", ""
    if ckpt_path is None or not Path(ckpt_path).exists():
        return "", "错误：请先选择模型权重文件。", ""

    try:
        set_seed(42)
        model, msg_len, device = load_model(ckpt_path)
        patch_size = _model_patch_size(model)

        img = _to_rgb_image(image)
        patches, _, _, _, _ = _image_to_patches(img, patch_size)

        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        all_bits = []
        for patch in patches:
            tensor = tf(patch).unsqueeze(0).to(device)
            decoded = model.encoder_decoder.decoder(tensor)
            compressed, _ = model._compress_message(decoded)
            bits = (compressed.cpu().numpy().round().clip(0, 1)).astype(int).flatten()
            all_bits.append(bits)

        all_bits_arr = np.array(all_bits)
        majority = (all_bits_arr.mean(axis=0) >= 0.5).astype(int)

        bits_str = _bits_to_bit_string(majority)

        agreement = (all_bits_arr == majority).mean(axis=0)
        info = f"patch 尺寸：{patch_size} x {patch_size}\n"
        info += f"总分块数：{all_bits_arr.shape[0]}\n"
        info += "图像来源：当前图片组件\n"
        info += f"位一致性：最低 {agreement.min():.2%}，平均 {agreement.mean():.2%}"
        if expected_bits.strip():
            target_bits, was_truncated = _normalize_bit_string(expected_bits, msg_len)
            info += "\n" + _bit_eval_report(target_bits, bits_str)
            if was_truncated:
                info += "\n原始输入超过 payload 长度，评估时已按模型长度截断。"

        return bits_str, info, bits_str

    except Exception as e:
        return "", f"提取过程出错：{str(e)}", ""


def apply_attack(
    image: np.ndarray,
    attack_type: str,
    attack_strength: float,
    cover_image: Optional[np.ndarray],
    ckpt_path: Optional[str],
) -> tuple[Optional[np.ndarray], str, Optional[np.ndarray]]:
    if image is None:
        return None, "错误：请先提供含水印图片。", None

    try:
        set_seed(42)
        img = _to_rgb_image(image)
        cfg = _load_checkpoint_config(ckpt_path)
        patch_size = _patch_size_from_config(cfg)
        device = resolve_device("auto")
        layer, noise_name, desc = _build_frontend_noise_layer(attack_type, attack_strength, cfg, device)

        if cover_image is None and noise_name in {"dropout", "cropout"}:
            return (
                None,
                "错误：该训练噪声层需要原图 cover 参与计算。请先在“水印嵌入”页生成含水印图片，再执行该攻击。",
                None,
            )

        cover = _to_rgb_image(cover_image) if cover_image is not None else img
        result = _apply_training_noise_to_image(img, cover, layer, device, patch_size)

        result_np = _pil_to_np(result)
        return result_np, f"攻击完成：训练噪声层模拟 {desc}；按 {patch_size} x {patch_size} 分块执行。", result_np

    except Exception as e:
        return None, f"攻击过程出错：{str(e)}", None


def create_app() -> gr.Blocks:
    with gr.Blocks(title="隐形数字水印系统") as app:
        gr.Markdown(
            """
            # 基于深度学习的隐形数字水印系统

            本系统支持将隐形水印嵌入图片中，并可直接对嵌入结果进行提取和常见攻击演示。
            """
        )

        with gr.Row():
            ckpt_file = gr.File(
                label="选择模型权重文件 (.pth)",
                file_types=[".pth"],
                type="filepath",
            )
            attack_cover_state = gr.State(value=None)

        with gr.Tabs():
            with gr.TabItem("水印嵌入"):
                with gr.Row():
                    with gr.Column():
                        embed_input = gr.Image(label="原始图片", type="numpy")
                        embed_message = gr.Textbox(
                            label="二进制水印",
                            placeholder="请输入 0/1，例如 110 或 011000010110001001100011",
                            max_lines=1,
                        )
                        embed_inference_mode = gr.Radio(
                            label="大图嵌入方式",
                            choices=["重叠融合（减轻方格）", "标准分块（准确率优先）"],
                            value="重叠融合（减轻方格）",
                        )
                        embed_btn = gr.Button("嵌入水印", variant="primary")

                    with gr.Column():
                        embed_output = gr.Image(label="含水印图片", type="numpy")
                        embed_info = gr.Textbox(label="处理状态", lines=5, interactive=False)

            with gr.TabItem("攻击演示"):
                with gr.Row():
                    with gr.Column():
                        attack_input = gr.Image(label="待攻击图片", type="numpy")
                        attack_type = gr.Dropdown(
                            label="攻击类型",
                            choices=[
                                "无攻击",
                                "JPEG 压缩模拟",
                                "微信压缩模拟",
                                "颜色量化模拟",
                                "高斯噪声",
                                "高斯模糊",
                                "缩放",
                                "随机裁剪",
                                "Dropout 遮挡",
                                "Cropout 保留区域",
                            ],
                            value="JPEG 压缩模拟",
                        )
                        attack_strength = gr.Slider(
                            0,
                            1,
                            value=0,
                            step=1,
                            label="使用训练配置中的 DCT/YUV JPEG 近似参数",
                            visible=False,
                        )
                        attack_btn = gr.Button("执行攻击", variant="primary")

                    with gr.Column():
                        attack_output = gr.Image(label="攻击后的图片", type="numpy")
                        attack_info = gr.Textbox(label="攻击状态", lines=3, interactive=False)
                        attack_expected_bits = gr.Textbox(
                            label="原始二进制水印（可选，用于计算 bit_acc / BER）",
                            placeholder="请输入嵌入时使用的 0/1 水印",
                            max_lines=1,
                        )
                        extract_attacked_btn = gr.Button("从攻击结果提取水印")
                        attack_extract_message = gr.Textbox(label="提取的二进制水印", interactive=False)
                        attack_extract_bits = gr.Textbox(label="提取的二进制位", interactive=False)
                        attack_extract_info = gr.Textbox(label="统计信息", lines=8, interactive=False)

            with gr.TabItem("水印提取"):
                with gr.Row():
                    with gr.Column():
                        extract_input = gr.Image(label="待提取图片", type="numpy")
                        extract_expected_bits = gr.Textbox(
                            label="原始二进制水印（可选，用于计算 bit_acc / BER）",
                            placeholder="请输入嵌入时使用的 0/1 水印",
                            max_lines=1,
                        )
                        extract_btn = gr.Button("提取水印", variant="primary")

                    with gr.Column():
                        extract_message = gr.Textbox(label="提取的二进制水印", interactive=False)
                        extract_bits = gr.Textbox(label="提取的二进制位", interactive=False)
                        extract_info = gr.Textbox(label="统计信息", lines=8, interactive=False)

            with gr.TabItem("使用说明"):
                gr.Markdown(
                    """
                    ## 工作原理

                    ### 基于权重的推理流程
                    1. 选择 `.pth` checkpoint 后，界面按 `configs/base.yaml` 构建 `HiddenSystem`。
                    2. `load_checkpoint()` 将权重加载到 encoder、decoder、discriminator 等模块。
                    3. 嵌入时，图片会按 checkpoint 中的训练图像尺寸切块，输入的二进制水印被整理为固定长度 bit tensor。
                    4. 标准分块模式会直接拼接图像块；重叠融合模式会使用滑窗和加权平均缓解大图方格边界。
                    5. 提取时，图片同样切块，每块经过 `decoder` 输出 bit，最后对所有图像块做多数投票。

                    ### 展示流程
                    - 嵌入完成后，结果会自动同步到“水印提取”和“攻击演示”，无需先保存再重新上传。
                    - 也可以在“水印提取”中直接上传已有含水印图片并提取，不要求先执行嵌入。
                    - “攻击演示”默认调用训练时的 `src/noise` 噪声层，并按 checkpoint 对应的训练 patch 尺寸分块执行，用于展示训练分布内的鲁棒性。
                    - 微信压缩为近似噪声层，不等同于真实微信客户端的完整压缩链路。
                    - 攻击后的结果会显示在当前页，也会自动同步到“水印提取”，可在当前页直接提取攻击结果。
                    """
                )

        embed_btn.click(
            fn=embed_watermark,
            inputs=[embed_input, embed_message, ckpt_file, embed_inference_mode],
            outputs=[
                embed_output,
                embed_info,
                extract_input,
                attack_input,
                extract_expected_bits,
                attack_expected_bits,
                attack_cover_state,
            ],
        )
        attack_type.change(
            fn=_attack_strength_update,
            inputs=[attack_type],
            outputs=[attack_strength],
        )
        attack_btn.click(
            fn=apply_attack,
            inputs=[
                attack_input,
                attack_type,
                attack_strength,
                attack_cover_state,
                ckpt_file,
            ],
            outputs=[attack_output, attack_info, extract_input],
        )
        extract_btn.click(
            fn=extract_watermark,
            inputs=[extract_input, ckpt_file, extract_expected_bits],
            outputs=[extract_message, extract_info, extract_bits],
        )
        extract_attacked_btn.click(
            fn=extract_watermark,
            inputs=[attack_output, ckpt_file, attack_expected_bits],
            outputs=[attack_extract_message, attack_extract_info, attack_extract_bits],
        )

        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #666;">
                毕业设计：基于深度学习的隐形数字水印系统
            </div>
            """
        )

    return app


def launch_app(share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860):
    app = create_app()
    app.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
    )


if __name__ == "__main__":
    launch_app()
