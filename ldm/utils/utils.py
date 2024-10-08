import os
import sys
import logging
import importlib
from logging.handlers import RotatingFileHandler
from omegaconf import OmegaConf

log_dir = "ldm_logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "ml_project.log")


def setup_logger(
    log_file, level=logging.INFO, max_bytes=5 * 1024 * 1024, backup_count=5
):
    """
    Set up a logger that logs messages to both a file and the console.

    Args:
        log_file (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        max_bytes (int): Maximum size of the log file before it is rotated (default: 5 MB).
        backup_count (int): Number of backup files to keep after rotation.

    Returns:
        logger: Configured logger instance.
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logger(log_file_path)


def conf_access(conf, conf_path: str):
    if "." in conf_path:
        conf_path = conf_path.split(".")
        for path in conf_path:
            conf = conf[path]

    return conf


def load_config(conf_path: str, attribute: str):
    try:
        conf = OmegaConf.load(conf_path)
        conf = conf_access(conf, attribute)
        params = conf.params

        return params
    except Exception as e:
        logger.error(f"Error loading config: {e}")
