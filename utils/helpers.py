"""Shared utility functions."""

import logging
import random
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible CPU and CUDA execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(level=logging.INFO, log_file=None):
    """Configure console logging and optionally write the same log to a file."""
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s — %(levelname)s — %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)
