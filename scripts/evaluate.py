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
from src.data.loaders import build_dataloaders
from src.engine.checkpoint import find_latest_checkpoint, load_checkpoint
from src.engine.evaluator import Evaluator
from src.models.hidden_system import HiddenSystem
from src.noise.manager import NoiseManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate refactored HiDDeN model.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. If omitted, latest in --run_dir/checkpoints is used.")
    parser.add_argument("--run_dir", default=None, help="Run folder used when --checkpoint is omitted.")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values, e.g. dataset.val_dir=data/val",
    )
    parser.add_argument("--sample_count", type=int, default=8, help="Number of samples to visualize.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_cfg = load_config(args.config)
    cfg = apply_overrides(base_cfg, args.override)

    device = resolve_device(cfg["experiment"].get("device", "auto"))

    eval_dir = create_run_directory(
        output_root=cfg["experiment"]["output_root"],
        experiment_name=f"{cfg['experiment']['name']}_eval",
    )
    logger = setup_logging(eval_dir / "logs" / "evaluate.log", level=cfg.get("logging", {}).get("level", "INFO"))
    save_config(cfg, eval_dir / "resolved_config.yaml")

    _, val_loader = build_dataloaders(cfg["dataset"], batch_size=int(cfg["train"]["batch_size"]))

    image_size = tuple(cfg["dataset"].get("image_size", [64, 64]))
    noise_manager = NoiseManager(cfg["noise"], device=device).to(device)
    model = HiddenSystem(
        model_cfg=cfg["model"],
        train_cfg=cfg["train"],
        image_size=(int(image_size[0]), int(image_size[1])),
        noise_manager=noise_manager,
        device=device,
    ).to(device)

    checkpoint = args.checkpoint
    if checkpoint is None:
        if args.run_dir is None:
            raise ValueError("Either --checkpoint or --run_dir must be provided.")
        latest = find_latest_checkpoint(f"{args.run_dir}/checkpoints")
        if latest is None:
            raise FileNotFoundError(f"No checkpoint found in {args.run_dir}/checkpoints")
        checkpoint = str(latest)

    payload = load_checkpoint(checkpoint, model=model, device=device, scaler=None, strict=False)
    epoch = int(payload.get("epoch", 0))
    logger.info("Loaded checkpoint: %s (epoch %s)", checkpoint, epoch)

    evaluator = Evaluator(model=model, device=device, message_length=int(cfg["model"]["message_length"]))
    metrics = evaluator.validate(val_loader, epoch=epoch)
    evaluator.save_visual_examples(val_loader, output_dir=eval_dir / "images", epoch=epoch, sample_count=args.sample_count)

    logger.info("Evaluation metrics: %s", metrics)
    logger.info("Evaluation artifacts saved to: %s", eval_dir)


if __name__ == "__main__":
    main()

