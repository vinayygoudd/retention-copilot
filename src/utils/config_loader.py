"""Configuration and path management utilities for Retention Copilot."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


def get_project_root() -> Path:
    """Return the absolute path to the `retention_copilot` project root directory."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load YAML configuration file and merge with environment overrides."""
    load_dotenv(get_project_root() / ".env")
    if config_path is None:
        env_path = os.getenv("RETENTION_CONFIG_PATH")
        if env_path:
            resolved = Path(env_path)
            if not resolved.is_absolute():
                resolved = get_project_root() / resolved
        else:
            resolved = get_project_root() / "configs" / "config.yaml"
    else:
        resolved = Path(config_path)
        if not resolved.is_absolute():
            resolved = get_project_root() / resolved

    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found at: {resolved}")

    with open(resolved, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    return config


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a structured console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
