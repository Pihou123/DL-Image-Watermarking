from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch


def resolve_device(device_name: str) -> torch.device:
    name = device_name.lower()
    if name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif name == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(name)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    return device


def create_run_directory(output_root: str, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_root) / f"{experiment_name}_{timestamp}"
    for sub in ("checkpoints", "images", "logs", "plots"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir
