"""Inference script: loads a model saved by train.py and writes predictions.

Usage:
    python predict.py
    python predict.py --test_path data/hidden_test.csv \
        --model_path model.pkl --output_path predictions.csv
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from common import DEFAULT_MODEL_PATH, ensure_parent_dir, load_csv
from models import BaseModel

DEFAULT_TEST_PATH = "data/hidden_test.csv"
DEFAULT_OUTPUT_PATH = "predictions.csv"

PREDICTION_COL = "prediction"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate predictions using a saved model."
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=DEFAULT_TEST_PATH,
        help="Path to the test CSV file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved model (joblib).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the predictions CSV.",
    )
    return parser.parse_args()


def load_model(path: str) -> BaseModel:
    """Load a model saved by train.py."""
    model = joblib.load(path)
    if not isinstance(model, BaseModel):
        raise TypeError(f"'{path}' is not a model saved by train.py")
    logger.info(f"Loaded '{path}': {type(model).__name__}")
    return model


def save_predictions(preds: np.ndarray, path: str) -> None:
    """Save predictions to CSV (single ``prediction`` column)."""
    ensure_parent_dir(path)
    pd.DataFrame({PREDICTION_COL: preds}).to_csv(path, index=False)
    logger.info(f"Saved {len(preds)} predictions to '{path}'")


def main() -> None:
    """Load data and model, predict, save results."""
    args = parse_args()
    test_df = load_csv(args.test_path)
    model = load_model(args.model_path)
    save_predictions(model.predict(test_df), args.output_path)


if __name__ == "__main__":
    main()
