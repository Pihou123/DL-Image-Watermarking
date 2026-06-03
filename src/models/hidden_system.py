"""隐形水印系统的训练、验证和推理封装。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import amp

from .discriminator import Discriminator
from .encoder_decoder import EncoderDecoder
from .vgg_loss import VGGLoss
from ..engine.metrics import compute_psnr, compute_ssim


class HiddenSystem(nn.Module):
    """组合编码器、解码器、判别器和损失函数。"""
    def __init__(self, model_cfg: dict, train_cfg: dict, image_size: tuple[int, int], noise_manager, device: torch.device):
        super().__init__()
        self.device = device
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        self.payload_length = int(model_cfg.get("payload_length", model_cfg["message_length"]))
        self.message_length = int(model_cfg["message_length"])

        if self.message_length != self.payload_length:
            assert self.message_length % self.payload_length == 0, \
                f"message_length ({self.message_length}) must be multiple of payload_length ({self.payload_length})"
            self.repeat_factor = self.message_length // self.payload_length
        else:
            self.repeat_factor = 1

        self.encoder_decoder = EncoderDecoder(model_cfg=model_cfg, image_size=image_size, noise_manager=noise_manager).to(device)
        self.discriminator = Discriminator(model_cfg=model_cfg).to(device)

        self.use_discriminator = bool(model_cfg.get("use_discriminator", True))
        self.use_adversarial_noise_training = bool(model_cfg.get("use_adversarial_noise_training", False))
        self.loss_weights = model_cfg.get("loss_weights", {"encoder": 1.0, "decoder": 1.0, "adversarial": 1.0})

        self.optimizer_encoder_decoder = torch.optim.Adam(
            self.encoder_decoder.parameters(), lr=float(train_cfg.get("lr_encoder_decoder", 1e-3))
        )
        self.optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr=float(train_cfg.get("lr_discriminator", 1e-3))
        )
        adversarial_noise_params = self.encoder_decoder.noise_manager.adversarial_parameters()
        self.optimizer_adversarial_noise = (
            torch.optim.Adam(adversarial_noise_params, lr=float(train_cfg.get("lr_adversarial_noise", train_cfg.get("lr_encoder_decoder", 1e-3))))
            if self.use_adversarial_noise_training and adversarial_noise_params
            else None
        )

        self.use_vgg = bool(model_cfg.get("use_vgg", False))
        self.vgg_loss_mode = str(model_cfg.get("vgg_loss_mode", "vgg")).lower()
        if self.vgg_loss_mode not in {"pixel", "vgg", "mixed"}:
            raise ValueError("model.vgg_loss_mode must be one of: pixel, vgg, mixed.")
        self.vgg_loss_weight = float(model_cfg.get("vgg_loss_weight", 1.0))
        self.vgg_loss = VGGLoss(device=device) if self.use_vgg and self.vgg_loss_mode != "pixel" else None
        self.use_yuv_loss = bool(model_cfg.get("use_yuv_loss", False))
        self.yuv_loss_weight = float(model_cfg.get("yuv_loss_weight", 1.0))
        yuv_weights = model_cfg.get("yuv_channel_weights", [1.0, 1.0, 1.0])
        if len(yuv_weights) != 3:
            raise ValueError("model.yuv_channel_weights must contain exactly 3 values: [Y, U, V].")
        self.register_buffer(
            "yuv_channel_weights",
            torch.tensor(yuv_weights, dtype=torch.float32).view(1, 3, 1, 1),
        )

        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.cover_label = 1.0
        self.encoded_label = 0.0

    @staticmethod
    def _set_requires_grad(parameters, requires_grad: bool) -> list[bool]:
        params = list(parameters)
        previous = [param.requires_grad for param in params]
        for param in params:
            param.requires_grad_(requires_grad)
        return previous

    @staticmethod
    def _restore_requires_grad(parameters, previous: list[bool]) -> None:
        for param, requires_grad in zip(parameters, previous):
            param.requires_grad_(requires_grad)

    def _expand_message(self, messages: torch.Tensor) -> torch.Tensor:
        """按重复因子扩展 payload bit。"""
        if self.repeat_factor == 1:
            return messages
        return messages.repeat_interleave(self.repeat_factor, dim=1)

    def _compress_message(self, decoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """对重复 bit 求平均，还原有效 payload。"""
        if self.repeat_factor == 1:
            return decoded, decoded
        # 对每组重复 bit 求平均。
        b = decoded.shape[0]
        reshaped = decoded.view(b, self.payload_length, self.repeat_factor)
        compressed = reshaped.mean(dim=2)
        return compressed, decoded

    def infer(self, images: torch.Tensor, messages: torch.Tensor, epoch: int | None = None):
        expanded = self._expand_message(messages)
        return self.encoder_decoder(images, expanded, epoch=epoch)

    def _encoder_image_loss(self, encoded_images: torch.Tensor, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        rgb_loss = self.mse(encoded_images, images)
        vgg_feature_loss = torch.tensor(0.0, device=encoded_images.device)
        base_loss = rgb_loss
        base_name = "rgb_mse"

        if self.vgg_loss is not None:
            encoded_features = self.vgg_loss(encoded_images)
            with torch.no_grad():
                cover_features = self.vgg_loss(images)
            vgg_feature_loss = self.mse(encoded_features, cover_features)

            if self.vgg_loss_mode == "vgg":
                base_loss = vgg_feature_loss
                base_name = "vgg_mse"
            elif self.vgg_loss_mode == "mixed":
                base_loss = rgb_loss + self.vgg_loss_weight * vgg_feature_loss
                base_name = "mixed_rgb_vgg_mse"

        yuv_loss = torch.tensor(0.0, device=encoded_images.device)
        if self.use_yuv_loss:
            encoded_yuv = self._rgb_to_yuv(encoded_images)
            cover_yuv = self._rgb_to_yuv(images)
            yuv_diff = (encoded_yuv - cover_yuv).pow(2)
            weights = self.yuv_channel_weights.to(device=encoded_images.device, dtype=yuv_diff.dtype)
            yuv_loss = (yuv_diff * weights).mean()

        total = base_loss + self.yuv_loss_weight * yuv_loss
        metrics = {
            "encoder_base_mse": float(base_loss.detach().item()),
            "encoder_rgb_mse": float(rgb_loss.detach().item()),
            "encoder_vgg_mse": float(vgg_feature_loss.detach().item()),
            "encoder_yuv_mse": float(yuv_loss.detach().item()),
            "encoder_loss_type": base_name,
        }
        return total, metrics

    @staticmethod
    def _rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
        # 将 [-1, 1] 转为 [0, 1]。
        x = (image + 1.0) / 2.0
        r = x[:, 0:1]
        g = x[:, 1:2]
        b = x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r - 0.51499 * g - 0.10001 * b
        return torch.cat([y, u, v], dim=1)

    def train_step(
        self,
        images: torch.Tensor,
        messages: torch.Tensor,
        scaler: amp.GradScaler | None,
        grad_clip_norm: float = 0.0,
        epoch: int | None = None,
    ) -> dict[str, float]:
        batch_size = images.shape[0]
        autocast_enabled = scaler is not None and self.device.type == "cuda"

        expanded_messages = self._expand_message(messages)

        self.encoder_decoder.train()
        self.discriminator.train()

        d_loss_cover = torch.tensor(0.0, device=self.device)
        d_loss_encoded = torch.tensor(0.0, device=self.device)
        g_loss_adv = torch.tensor(0.0, device=self.device)
        adv_noise_loss = torch.tensor(0.0, device=self.device)
        payload_weight = float(self.loss_weights.get("payload", float(self.loss_weights.get("decoder", 1.0))))

        if self.use_discriminator:
            self.optimizer_discriminator.zero_grad(set_to_none=True)

            with amp.autocast(device_type="cuda", enabled=autocast_enabled):
                encoded_images, _, _, _ = self.encoder_decoder(images, expanded_messages, epoch=epoch)
                cover_target = torch.full((batch_size, 1), self.cover_label, device=self.device)
                encoded_target = torch.full((batch_size, 1), self.encoded_label, device=self.device)

                d_on_cover = self.discriminator(images)
                d_on_encoded = self.discriminator(encoded_images.detach())
                d_loss_cover = self.bce(d_on_cover, cover_target)
                d_loss_encoded = self.bce(d_on_encoded, encoded_target)
                d_loss_total = d_loss_cover + d_loss_encoded

            if scaler is not None:
                scaler.scale(d_loss_total).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(self.optimizer_discriminator)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=grad_clip_norm)
                scaler.step(self.optimizer_discriminator)
            else:
                d_loss_total.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=grad_clip_norm)
                self.optimizer_discriminator.step()

        if self.optimizer_adversarial_noise is not None:
            self.optimizer_adversarial_noise.zero_grad(set_to_none=True)
            encoder_decoder_params = list(self.encoder_decoder.encoder.parameters()) + list(self.encoder_decoder.decoder.parameters())
            previous = self._set_requires_grad(encoder_decoder_params, False)
            try:
                with amp.autocast(device_type="cuda", enabled=autocast_enabled):
                    with torch.no_grad():
                        encoded_for_attack = self.encoder_decoder.encode(images, expanded_messages)
                    noised_for_attack = self.encoder_decoder.noise_manager.apply_adversarial(
                        encoded_for_attack.detach(),
                        images,
                    )
                    decoded_for_attack = self.encoder_decoder.decoder(noised_for_attack)
                    compressed_for_attack, _ = self._compress_message(decoded_for_attack)
                    adv_noise_loss = (
                        self.mse(decoded_for_attack, expanded_messages)
                        + payload_weight * self.mse(compressed_for_attack, messages)
                    )
                    adversary_objective = -adv_noise_loss

                if scaler is not None:
                    scaler.scale(adversary_objective).backward()
                    scaler.step(self.optimizer_adversarial_noise)
                    scaler.update()
                else:
                    adversary_objective.backward()
                    self.optimizer_adversarial_noise.step()
            finally:
                self._restore_requires_grad(encoder_decoder_params, previous)

        self.optimizer_encoder_decoder.zero_grad(set_to_none=True)
        adversarial_params = self.encoder_decoder.noise_manager.adversarial_parameters()
        previous_adversarial = self._set_requires_grad(adversarial_params, False)
        try:
            with amp.autocast(device_type="cuda", enabled=autocast_enabled):
                encoded_images, noised_images, decoded_messages, _ = self.encoder_decoder(images, expanded_messages, epoch=epoch)

                if self.use_discriminator:
                    target_encoded_as_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
                    d_on_encoded_for_gen = self.discriminator(encoded_images)
                    g_loss_adv = self.bce(d_on_encoded_for_gen, target_encoded_as_cover)
                else:
                    g_loss_adv = torch.tensor(0.0, device=self.device)

                g_loss_enc, encoder_loss_metrics = self._encoder_image_loss(encoded_images, images)

                g_loss_dec = self.mse(decoded_messages, expanded_messages)
                compressed_decoded, _ = self._compress_message(decoded_messages)
                g_loss_payload = self.mse(compressed_decoded, messages)
                total_loss = (
                    float(self.loss_weights.get("adversarial", 1.0)) * g_loss_adv
                    + float(self.loss_weights.get("encoder", 1.0)) * g_loss_enc
                    + float(self.loss_weights.get("decoder", 1.0)) * g_loss_dec
                    + payload_weight * g_loss_payload
                )

            if scaler is not None:
                scaler.scale(total_loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(self.optimizer_encoder_decoder)
                    torch.nn.utils.clip_grad_norm_(self.encoder_decoder.parameters(), max_norm=grad_clip_norm)
                scaler.step(self.optimizer_encoder_decoder)
                scaler.update()
            else:
                total_loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.encoder_decoder.parameters(), max_norm=grad_clip_norm)
                self.optimizer_encoder_decoder.step()
        finally:
            self._restore_requires_grad(adversarial_params, previous_adversarial)

        bit_error, bit_acc = self._bit_metrics(compressed_decoded, messages)
        psnr_val = compute_psnr(encoded_images, images)
        ssim_val = compute_ssim(encoded_images, images)
        return {
            "loss": float(total_loss.detach().item()),
            "encoder_mse": float(g_loss_enc.detach().item()),
            "encoder_base_mse": encoder_loss_metrics["encoder_base_mse"],
            "encoder_rgb_mse": encoder_loss_metrics["encoder_rgb_mse"],
            "encoder_vgg_mse": encoder_loss_metrics["encoder_vgg_mse"],
            "encoder_yuv_mse": encoder_loss_metrics["encoder_yuv_mse"],
            "decoder_mse": float(g_loss_dec.detach().item()),
            "adversarial_bce": float(g_loss_adv.detach().item()),
            "discr_cover_bce": float(d_loss_cover.detach().item()),
            "discr_encoded_bce": float(d_loss_encoded.detach().item()),
            "adversarial_noise_mse": float(adv_noise_loss.detach().item()),
            "bit_error": float(bit_error),
            "bit_acc": float(bit_acc),
            "psnr": psnr_val,
            "ssim": ssim_val,
        }

    @torch.no_grad()
    def validate_step(self, images: torch.Tensor, messages: torch.Tensor, epoch: int | None = None) -> dict[str, float]:
        batch_size = images.shape[0]

        expanded_messages = self._expand_message(messages)

        self.encoder_decoder.eval()
        self.discriminator.eval()

        encoded_images, _, decoded_messages, _ = self.encoder_decoder(images, expanded_messages, epoch=epoch)

        if self.use_discriminator:
            cover_target = torch.full((batch_size, 1), self.cover_label, device=self.device)
            encoded_target = torch.full((batch_size, 1), self.encoded_label, device=self.device)

            d_on_cover = self.discriminator(images)
            d_on_encoded = self.discriminator(encoded_images)
            d_loss_cover = self.bce(d_on_cover, cover_target)
            d_loss_encoded = self.bce(d_on_encoded, encoded_target)
            g_loss_adv = self.bce(self.discriminator(encoded_images), cover_target)
        else:
            d_loss_cover = torch.tensor(0.0, device=self.device)
            d_loss_encoded = torch.tensor(0.0, device=self.device)
            g_loss_adv = torch.tensor(0.0, device=self.device)

        g_loss_enc, encoder_loss_metrics = self._encoder_image_loss(encoded_images, images)

        g_loss_dec = self.mse(decoded_messages, expanded_messages)
        compressed_decoded, _ = self._compress_message(decoded_messages)
        g_loss_payload = self.mse(compressed_decoded, messages)
        payload_weight = float(self.loss_weights.get("payload", float(self.loss_weights.get("decoder", 1.0))))
        total_loss = (
            float(self.loss_weights.get("adversarial", 1.0)) * g_loss_adv
            + float(self.loss_weights.get("encoder", 1.0)) * g_loss_enc
            + float(self.loss_weights.get("decoder", 1.0)) * g_loss_dec
            + payload_weight * g_loss_payload
        )

        bit_error, bit_acc = self._bit_metrics(compressed_decoded, messages)
        psnr_val = compute_psnr(encoded_images, images)
        ssim_val = compute_ssim(encoded_images, images)
        return {
            "loss": float(total_loss.detach().item()),
            "encoder_mse": float(g_loss_enc.detach().item()),
            "encoder_base_mse": encoder_loss_metrics["encoder_base_mse"],
            "encoder_rgb_mse": encoder_loss_metrics["encoder_rgb_mse"],
            "encoder_vgg_mse": encoder_loss_metrics["encoder_vgg_mse"],
            "encoder_yuv_mse": encoder_loss_metrics["encoder_yuv_mse"],
            "decoder_mse": float(g_loss_dec.detach().item()),
            "adversarial_bce": float(g_loss_adv.detach().item()),
            "discr_cover_bce": float(d_loss_cover.detach().item()),
            "discr_encoded_bce": float(d_loss_encoded.detach().item()),
            "bit_error": float(bit_error),
            "bit_acc": float(bit_acc),
            "psnr": psnr_val,
            "ssim": ssim_val,
        }

    @staticmethod
    def _bit_metrics(decoded_messages: torch.Tensor, messages: torch.Tensor) -> tuple[float, float]:
        decoded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        targets = messages.detach().cpu().numpy()
        bit_error = float(np.mean(np.abs(decoded - targets)))
        bit_acc = float(1.0 - bit_error)
        return bit_error, bit_acc
