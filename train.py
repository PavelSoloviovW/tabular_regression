"""Training script: picks the first suitable model strategy and fits it.

Strategies are tried in order: formula first, RandomForest as the fallback.

Usage:
    python train.py
    python train.py --train_path data/train.csv --model_path model.pkl --threshold 1e-3
"""

from __future__ import annotations

import argparse

import joblib
from loguru import logger

from common import DEFAULT_MODEL_PATH, ensure_parent_dir, load_csv
from models import BaseModel, FormulaModel, MLModel

DEFAULT_TRAIN_PATH = "data/train.csv"
DEFAULT_THRESHOLD = 1e-3


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a regression model (formula check with ML fallback)."
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default=DEFAULT_TRAIN_PATH,
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="RMSE threshold for accepting the formula.",
    )
    return parser.parse_args()


def save_model(model: BaseModel, model_path: str) -> None:
    """Save the model with joblib."""
    ensure_parent_dir(model_path)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to '{model_path}'")


def main() -> None:
    """Select the first suitable strategy, fit and save it."""
    args = parse_args()
    df = load_csv(args.train_path)

    strategies = [FormulaModel(args.threshold), MLModel()]
    model = next(m for m in strategies if m.is_suitable(df))
    logger.info(f"Selected strategy: {type(model).__name__}")

    model.fit(df)
    save_model(model, args.model_path)


if __name__ == "__main__":
    main()
