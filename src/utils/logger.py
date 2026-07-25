import logging
import logging.config
from pathlib import Path

import yaml

_CONFIG_LOADED = False

def _load_config(config_path: Path) -> None:
    global _CONFIG_LOADED
    if _CONFIG_LOADED:
        return

    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f.read())
    
    logging.config.dictConfig(config)
    _CONFIG_LOADED = True

def setup_logger(
    name:str, 
    config_path: Path = "config/logging_config.yaml"
    ) -> logging.Logger:
    _load_config(Path(config_path))
    return logging.getLogger(name)