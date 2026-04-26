from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import shutil
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a run folder into a timestamped package.")
    parser.add_argument("--run_dir", required=True, help="Run directory to export.")
    parser.add_argument("--out_root", default="outputs/exports", help="Export root directory.")
    parser.add_argument("--name", default=None, help="Optional custom export name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    name = args.name or f"{run_dir.name}_{time.strftime('%Y%m%d-%H%M%S')}"
    export_dir = out_root / name

    shutil.copytree(run_dir, export_dir, dirs_exist_ok=False)
    print(export_dir)


if __name__ == "__main__":
    main()


