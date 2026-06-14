from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_CACHE: dict[str, Any] | None = None


def load_config() -> dict[str, Any]:
    global _CONFIG_CACHE

    if _CONFIG_CACHE is None:
        root = next(
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "configs" / "config.yaml").exists()
        )
        config_path = root / "configs" / "config.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            _CONFIG_CACHE = yaml.safe_load(handle) or {}
        _apply_mode_override(_CONFIG_CACHE)

    return _CONFIG_CACHE


def _apply_mode_override(config: dict[str, Any]) -> None:
    mode = os.environ.get("MARS_ACTIVE_MODE") or config.get("active_mode")
    modes = config.get("modes", {})
    if not mode or mode not in modes:
        return
    mode_cfg = modes[mode] or {}
    simulator = config.setdefault("simulator", {})
    scale = simulator.setdefault("scale", {})
    target_scale = simulator.setdefault("target_scale", {})
    if "products" in mode_cfg:
        scale["products"] = int(mode_cfg["products"])
        target_scale["products"] = int(mode_cfg["products"])
    if "users" in mode_cfg:
        scale["users"] = int(mode_cfg["users"])
        target_scale["users"] = int(mode_cfg["users"])
    if "events" in mode_cfg:
        scale["events"] = int(mode_cfg["events"])
        target_scale["events"] = int(mode_cfg["events"])
