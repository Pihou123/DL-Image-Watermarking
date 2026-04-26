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
    def __init__(self, model_cfg: dict, train_cfg: dict, image_size: tuple[int, int], noise_manager, device: torch.device):
        super().__init__()
        self.device = device
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        self.encoder_decoder = EncoderDecoder(model_cfg=model_cfg, image_size=image_size, noise_manager=noise_manager).to(device)
        self.discriminator = Discriminator(model_cfg=model_cfg).to(device)

        self.use_discriminator = bool(model_cfg.get("use_discriminator", True))
        self.loss_weights = model_cfg.get("loss_weights", {"encoder": 1.0, "decoder": 1.0, "adversarial": 1.0})

        self.optimizer_encoder_decoder = torch.optim.Adam(
            self.encoder_decoder.parameters(), lr=float(train_cfg.get("lr_encoder_decoder", 1e-3))
        )
        self.optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr=float(train_cfg.get("lr_discriminator", 1e-3))
        )

        self.use_vgg = bool(model_cfg.get("use_vgg", False))
        self.vgg_loss = VGGLoss(device=device) if self.use_vgg else None

        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.cover_label = 1.0
        self.encoded_label = 0.0

    def infer(self, images: torch.Tensor, messages: torch.Tensor, epoch: int | None = None):
        return self.encoder_decoder(images, messages, epoch=epoch)

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

        self.encoder_decoder.train()
        self.discriminator.train()

        d_loss_cover = torch.tensor(0.0, device=self.device)
        d_loss_encoded = torch.tensor(0.0, device=self.device)
        g_loss_adv = torch.tensor(0.0, device=self.device)

        if self.use_discriminator:
            self.optimizer_discriminator.zero_grad(set_to_none=True)

            with amp.autocast(device_type="cuda", enabled=autocast_enabled):
                encoded_images, _, _, _ = self.encoder_decoder(images, messages, epoch=epoch)
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

        self.optimizer_encoder_decoder.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", enabled=autocast_enabled):
            encoded_images, noised_images, decoded_messages, _ = self.encoder_decoder(images, messages, epoch=epoch)

            if self.use_discriminator:
                target_encoded_as_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
                d_on_encoded_for_gen = self.discriminator(encoded_images)
                g_loss_adv = self.bce(d_on_encoded_for_gen, target_encoded_as_cover)
            else:
                g_loss_adv = torch.tensor(0.0, device=self.device)

            if self.vgg_loss is None:
                g_loss_enc = self.mse(encoded_images, images)
            else:
                g_loss_enc = self.mse(self.vgg_loss(encoded_images), self.vgg_loss(images))

            g_loss_dec = self.mse(decoded_messages, messages)
            total_loss = (
                float(self.loss_weights.get("adversarial", 1.0)) * g_loss_adv
                + float(self.loss_weights.get("encoder", 1.0)) * g_loss_enc
                + float(self.loss_weights.get("decoder", 1.0)) * g_loss_dec
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

        bit_error, bit_acc = self._bit_metrics(decoded_messages, messages)
        psnr_val = compute_psnr(encoded_images, images)
        ssim_val = compute_ssim(encoded_images, images)
        return {
            "loss": float(total_loss.detach().item()),
            "encoder_mse": float(g_loss_enc.detach().item()),
            "decoder_mse": float(g_loss_dec.detach().item()),
            "adversarial_bce": float(g_loss_adv.detach().item()),
            "discr_cover_bce": float(d_loss_cover.detach().item()),
            "discr_encoded_bce": float(d_loss_encoded.detach().item()),
            "bit_error": float(bit_error),
            "bit_acc": float(bit_acc),
            "psnr": psnr_val,
            "ssim": ssim_val,
        }

    @torch.no_grad()
    def validate_step(self, images: torch.Tensor, messages: torch.Tensor, epoch: int | None = None) -> dict[str, float]:
        batch_size = images.shape[0]

        self.encoder_decoder.eval()
        self.discriminator.eval()

        encoded_images, _, decoded_messages, _ = self.encoder_decoder(images, messages, epoch=epoch)

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

        if self.vgg_loss is None:
            g_loss_enc = self.mse(encoded_images, images)
        else:
            g_loss_enc = self.mse(self.vgg_loss(encoded_images), self.vgg_loss(images))

        g_loss_dec = self.mse(decoded_messages, messages)
        total_loss = (
            float(self.loss_weights.get("adversarial", 1.0)) * g_loss_adv
            + float(self.loss_weights.get("encoder", 1.0)) * g_loss_enc
            + float(self.loss_weights.get("decoder", 1.0)) * g_loss_dec
        )

        bit_error, bit_acc = self._bit_metrics(decoded_messages, messages)
        psnr_val = compute_psnr(encoded_images, images)
        ssim_val = compute_ssim(encoded_images, images)
        return {
            "loss": float(total_loss.detach().item()),
            "encoder_mse": float(g_loss_enc.detach().item()),
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
