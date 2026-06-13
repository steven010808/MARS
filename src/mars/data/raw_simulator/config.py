from __future__ import annotations

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

    return _CONFIG_CACHE
