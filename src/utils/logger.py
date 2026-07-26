import logging
import logging.config
from pathlib import Path

import yaml

# /home/username/path/to/pearls-aqi-predictor
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "logging_config.yml"
LOGS_DIR = PROJECT_ROOT / "logs"

_CONFIG_LOADED = False


def _load_config(config_path: Path | None = None) -> None:
    global _CONFIG_LOADED
    if _CONFIG_LOADED:
        return

    LOGS_DIR.mkdir(exist_ok=True)

    config_path = config_path or CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f.read())

    logging.config.dictConfig(config)
    _CONFIG_LOADED = True


def setup_logger(name: str, config_path: Path | None = None) -> logging.Logger:
    _load_config(config_path)
    return logging.getLogger(name)