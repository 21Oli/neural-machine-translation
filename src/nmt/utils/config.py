"""YAML configuration loading and merging."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to a .yaml file.

    Returns:
        Configuration dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded from %s", path)
    return config or {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two configuration dictionaries.

    Values in `override` take precedence. Nested dicts are merged recursively;
    all other types are replaced.

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.

    Returns:
        Merged configuration dictionary.
    """
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = merge_configs(result[key], val)
        else:
            result[key] = val
    return result


def get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely retrieve a deeply nested config value.

    Args:
        config: Configuration dictionary.
        *keys: Sequence of keys to traverse.
        default: Value to return if the key path is not found.

    Returns:
        Value at the key path, or default.
    """
    current = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is default:
            return default
    return current
