import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_config(file: str | Path):
    file = Path(file)

    if not file.exists():
        raise ValueError(f"Cannot find config file at {file}")

    try:
        with open(file) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.exception(f"Error reading config file: {e}")
    
    return config