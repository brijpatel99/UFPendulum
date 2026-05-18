"""
config/config_loader.py

Typical usage:
    load specific configuration for a given experiment
"""

import tomllib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ======================================================================= #
# Config
# ======================================================================= #

def load_config(path: str) -> dict:
    """
    Load experiment configuration from a TOML file.

    Args:
        path : path to .toml config file

    Returns:
        dict of configuration values
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("Config file not found: %s" % path)
    with open(p, "rb") as f:
        cfg = tomllib.load(f)
    logger.info("Loaded config from %s", path)
    return cfg

