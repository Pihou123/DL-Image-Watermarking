from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from .metrics import MetricsAverager, compute_psnr, compute_ssim
from ..visualize.image_grid import save_difference_heatmap, save_triplet_grid


class Evaluator:
    def __init__(self, model, device: torch.device, message_length: int):
        self.model = model
        self.device = device
        self.message_length = message_length

    @torch.no_grad()
    def validate(self, val_loader, epoch: int) -> dict[str, float]:
        meter = MetricsAverager()
        progress = tqdm(val_loader, desc=f"Val {epoch}", leave=False)

        for images, _ in progress:
            images = images.to(self.device)
            messages = torch.randint(
                0,
                2,
                (images.shape[0], self.message_length),
                dtype=torch.float32,
                device=self.device,
            )
            metrics = self.model.validate_step(images, messages, epoch=epoch)
            meter.update(metrics)
            avg = meter.averages()
            progress.set_postfix(
                loss=f"{avg.get('loss', 0.0):.4f}",
                bit_acc=f"{avg.get('bit_acc', 0.0):.4f}",
                psnr=f"{avg.get('psnr', 0.0):.2f}",
                ssim=f"{avg.get('ssim', 0.0):.4f}",
            )

        return meter.averages()

    @torch.no_grad()
    def save_visual_examples(self, val_loader, output_dir: str | Path, epoch: int, sample_count: int = 8) -> None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        batch = next(iter(val_loader))
        images, _ = batch
        images = images[:sample_count].to(self.device)

        messages = torch.randint(0, 2, (images.shape[0], self.message_length), dtype=torch.float32, device=self.device)

        self.model.encoder_decoder.eval()
        encoded, noised, _, noise_meta = self.model.infer(images, messages, epoch=epoch)

        grid_path = out_dir / f"epoch_{epoch:04d}_triplet.png"
        diff_path = out_dir / f"epoch_{epoch:04d}_diff_heatmap.png"

        save_triplet_grid(images, encoded, noised, output_path=grid_path, max_count=sample_count)
        save_difference_heatmap(images, encoded, output_path=diff_path, max_count=sample_count)

        noise_info = out_dir / f"epoch_{epoch:04d}_noise.txt"
        psnr_val = compute_psnr(encoded, images)
        ssim_val = compute_ssim(encoded, images)
        noise_info.write_text(
            f"{noise_meta}\nPSNR={psnr_val:.2f} dB | SSIM={ssim_val:.4f}",
            encoding="utf-8",
        )
