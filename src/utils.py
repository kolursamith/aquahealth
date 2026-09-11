"""Shared helpers used across the training and inference pipeline."""

import logging
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (CPU, CUDA and MPS) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
