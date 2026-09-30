"""Load the public PGRAD YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load ``core_method/config.yaml`` unless another path is supplied."""
    path = Path(config_path) if config_path is not None else Path(__file__).with_name("config.yaml")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    return config
