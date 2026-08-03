"""
src/utils.py
------------
Module providing shared helper utilities for directory management, logger
initialization, and path constants across the retail sales analytics pipeline.

Author: Data Analytics Team
"""

import logging
import os
import sys
from typing import Union


# ---------------------------------------------------------------------------
# Project Paths Definition
# ---------------------------------------------------------------------------
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH: str = os.path.join(PROJECT_ROOT, "data", "raw", "sales_data.csv")
PROCESSED_DATA_PATH: str = os.path.join(PROJECT_ROOT, "data", "processed", "sales_data_clean.csv")
OUTPUTS_DIR: str = os.path.join(PROJECT_ROOT, "outputs")
GRAPHS_DIR: str = os.path.join(PROJECT_ROOT, "graphs")
REPORTS_DIR: str = os.path.join(PROJECT_ROOT, "reports")
MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")


def get_logger(name: str) -> logging.Logger:
    """
    Configure and return a dual-handler logger writing to stdout and pipeline.log.

    Avoids creating duplicate handlers if the logger has already been initialized.

    Parameters
    ----------
    name : str
        Name of the logger (typically __name__ of the calling module).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler
    log_file_path = os.path.join(log_dir, "pipeline.log")
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def ensure_dirs(*paths: Union[str, bytes, os.PathLike]) -> None:
    """
    Create one or more directories if they do not already exist.

    Parameters
    ----------
    *paths : Union[str, bytes, os.PathLike]
        One or many path-like objects to create.
    """
    for path in paths:
        if path:
            os.makedirs(path, exist_ok=True)
