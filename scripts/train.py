from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.common.config import apply_overrides, load_config, save_config
from src.common.logging_utils import setup_logging
from src.common.runtime import create_run_directory, resolve_device
from src.common.seed import set_seed
from src.data.loaders import build_dataloaders
from src.engine.checkpoint import load_checkpoint
from src.engine.trainer import Trainer
from src.models.hidden_system import HiddenSystem
from src.noise.manager import NoiseManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train refactored HiDDeN model.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values, e.g. train.epochs=50 noise.strategy=chain",
    )
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_cfg = load_config(args.config)
    cfg = apply_overrides(base_cfg, args.override)

    device = resolve_device(cfg["experiment"].get("device", "auto"))

    run_dir = create_run_directory(
        output_root=cfg["experiment"]["output_root"],
        experiment_name=cfg["experiment"]["name"],
    )

    logger = setup_logging(run_dir / "logs" / "train.log", level=cfg.get("logging", {}).get("level", "INFO"))
    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s", device)

    set_seed(int(cfg["experiment"].get("seed", 42)))
    save_config(cfg, run_dir / "resolved_config.yaml")

    train_loader, val_loader = build_dataloaders(cfg["dataset"], batch_size=int(cfg["train"]["batch_size"]))

    image_size = tuple(cfg["dataset"].get("image_size", [64, 64]))
    noise_manager = NoiseManager(cfg["noise"], device=device).to(device)
    model = HiddenSystem(
        model_cfg=cfg["model"],
        train_cfg=cfg["train"],
        image_size=(int(image_size[0]), int(image_size[1])),
        noise_manager=noise_manager,
        device=device,
    ).to(device)

    if args.resume:
        payload = load_checkpoint(args.resume, model=model, device=device, scaler=None, strict=False)
        resumed_epoch = int(payload.get("epoch", 0)) + 1
        cfg["train"]["start_epoch"] = resumed_epoch
        logger.info("Resumed from %s at epoch %s", args.resume, resumed_epoch)

    trainer = Trainer(model=model, config=cfg, device=device, run_dir=run_dir, logger=logger)
    trainer.fit(train_loader, val_loader)

    logger.info("Training completed.")
    logger.info("Artifacts saved to: %s", run_dir)


if __name__ == "__main__":
    main()


