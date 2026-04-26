from __future__ import annotations

from pathlib import Path

import torch
from torch import amp
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .evaluator import Evaluator
from .metrics import MetricsAverager
from ..visualize.curves import plot_history, write_history_csv


class Trainer:
    def __init__(self, model, config: dict, device: torch.device, run_dir: str | Path, logger):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger

        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.image_dir = self.run_dir / "images"
        self.plot_dir = self.run_dir / "plots"

        self.train_cfg = config["train"]
        self.model_cfg = config["model"]
        self.vis_cfg = config.get("visualization", {})

        self.message_length = int(self.model_cfg["message_length"])

        amp_enabled = bool(self.train_cfg.get("use_amp", False)) and self.device.type == "cuda"
        self.scaler = amp.GradScaler("cuda", enabled=amp_enabled)

        self.history: list[dict[str, float]] = []
        self.best_val_bit_acc = -1.0

    def fit(self, train_loader, val_loader) -> list[dict[str, float]]:
        start_epoch = int(self.train_cfg.get("start_epoch", 1))
        epochs = int(self.train_cfg.get("epochs", 10))
        validate_every = int(self.train_cfg.get("validate_every", 1))
        save_every = int(self.train_cfg.get("save_every", 1))

        evaluator = Evaluator(self.model, self.device, self.message_length)

        for epoch in range(start_epoch, epochs + 1):
            train_metrics = self._train_one_epoch(train_loader, epoch=epoch)

            val_metrics: dict[str, float] = {}
            if validate_every > 0 and epoch % validate_every == 0:
                val_metrics = evaluator.validate(val_loader, epoch=epoch)

            if int(self.vis_cfg.get("save_every", 1)) > 0 and epoch % int(self.vis_cfg.get("save_every", 1)) == 0:
                evaluator.save_visual_examples(
                    val_loader,
                    output_dir=self.image_dir,
                    epoch=epoch,
                    sample_count=int(self.vis_cfg.get("sample_count", 8)),
                )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss", 0.0),
                "train_bit_acc": train_metrics.get("bit_acc", 0.0),
                "train_bit_error": train_metrics.get("bit_error", 0.0),
                "train_psnr": train_metrics.get("psnr", float("nan")),
                "train_ssim": train_metrics.get("ssim", float("nan")),
                "val_loss": val_metrics.get("loss", float("nan")),
                "val_bit_acc": val_metrics.get("bit_acc", float("nan")),
                "val_bit_error": val_metrics.get("bit_error", float("nan")),
                "val_psnr": val_metrics.get("psnr", float("nan")),
                "val_ssim": val_metrics.get("ssim", float("nan")),
            }
            self.history.append(row)

            self._log_epoch_summary(epoch, train_metrics, val_metrics)

            if save_every > 0 and epoch % save_every == 0:
                ckpt_path = self.ckpt_dir / f"epoch_{epoch:04d}.pth"
                save_checkpoint(ckpt_path, epoch=epoch, model=self.model, config=self.config, scaler=self.scaler)

            val_bit_acc = val_metrics.get("bit_acc")
            if val_bit_acc is not None and val_bit_acc > self.best_val_bit_acc:
                self.best_val_bit_acc = float(val_bit_acc)
                save_checkpoint(self.ckpt_dir / "best.pth", epoch=epoch, model=self.model, config=self.config, scaler=self.scaler)

            write_history_csv(self.history, self.run_dir / "metrics.csv")
            plot_history(self.history, self.plot_dir / "training_curves.png")

        return self.history

    def _train_one_epoch(self, train_loader, epoch: int) -> dict[str, float]:
        meter = MetricsAverager()
        grad_clip_norm = float(self.train_cfg.get("grad_clip_norm", 0.0))
        log_interval = int(self.train_cfg.get("log_interval", 10))

        progress = tqdm(train_loader, desc=f"Train {epoch}", leave=False)

        for step, (images, _) in enumerate(progress, start=1):
            images = images.to(self.device)
            messages = torch.randint(
                0,
                2,
                (images.shape[0], self.message_length),
                dtype=torch.float32,
                device=self.device,
            )

            metrics = self.model.train_step(
                images,
                messages,
                scaler=self.scaler if self.scaler.is_enabled() else None,
                grad_clip_norm=grad_clip_norm,
                epoch=epoch,
            )
            meter.update(metrics)

            if step == 1 or step % log_interval == 0:
                avg = meter.averages()
                progress.set_postfix(
                    loss=f"{avg.get('loss', 0.0):.4f}",
                    dec=f"{avg.get('decoder_mse', 0.0):.4f}",
                    enc=f"{avg.get('encoder_mse', 0.0):.4f}",
                    bit_acc=f"{avg.get('bit_acc', 0.0):.4f}",
                    psnr=f"{avg.get('psnr', 0.0):.2f}",
                )

        return meter.averages()

    def _log_epoch_summary(self, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
        msg = (
            f"Epoch {epoch} | "
            f"train_loss={train_metrics.get('loss', 0.0):.4f} | "
            f"train_bit_acc={train_metrics.get('bit_acc', 0.0):.4f}"
        )
        if val_metrics:
            msg += (
                f" | val_loss={val_metrics.get('loss', 0.0):.4f}"
                f" | val_bit_acc={val_metrics.get('bit_acc', 0.0):.4f}"
                f" | val_psnr={val_metrics.get('psnr', 0.0):.2f}"
                f" | val_ssim={val_metrics.get('ssim', 0.0):.4f}"
            )
        self.logger.info(msg)
