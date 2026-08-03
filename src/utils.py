"""
utils.py
--------
Shared utility functions: logging setup, path helpers, and common
constants used across the analytics pipeline.
"""

import logging
import os
import sys

# ---------------------------------------------------------------------------
# Project paths (relative to the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "sales_data.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "sales_data_clean.csv")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
GRAPHS_DIR = os.path.join(PROJECT_ROOT, "graphs")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def get_logger(name: str) -> logging.Logger:
    """
    Create and configure a logger that writes to both console and a
    shared log file (logs/pipeline.log).

    Parameters
    ----------
    name : str
        Name of the logger, typically __name__ of the calling module.

    Returns
    -------
    logging.Logger
    """
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid adding duplicate handlers if the logger already exists
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(log_dir, "pipeline.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def ensure_dirs(*paths):
    """Create directories if they do not already exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)
