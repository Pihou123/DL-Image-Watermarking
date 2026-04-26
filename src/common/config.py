from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a dictionary.")
    return cfg


def save_config(config: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    cfg = deepcopy(config)
    if not overrides:
        return cfg

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is invalid. Use key=value format.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        value = _parse_literal(raw_value.strip())
        _set_nested_value(cfg, key, value)
    return cfg


def _parse_literal(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _set_nested_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value
