"""Shared helpers used across the training and inference pipeline."""

import logging
import random

import torch


def set_seed(seed: int) -> None:
    """Seed python, torch (all devices), and numpy when it is installed.

    numpy is seeded opportunistically because it is not a declared dependency
    until the data layers; once it is, the guard can go.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
