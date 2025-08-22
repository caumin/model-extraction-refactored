"""Module for loading experiment configurations from YAML files."""

import yaml
from pathlib import Path

def load_config(config_path: str | Path) -> dict:
    """Loads a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[DEBUG] Loaded config: {config}")
    return config