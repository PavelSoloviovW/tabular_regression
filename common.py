"""Shared constants and helpers for train.py and predict.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

DEFAULT_MODEL_PATH = "./models/product_model.pkl"


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file and log its size."""
    df = pd.read_csv(path)
    logger.info(f"Loaded '{path}': {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory of ``path`` if needed."""
    parent = Path(path).parent
    if parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)
