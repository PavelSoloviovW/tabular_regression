"""Model strategies: a formula and a RandomForest fallback.

Every model implements the same interface:
    is_suitable(df) - does this strategy fit the dataset?
    fit(df)         - train on the dataset
    predict(df)     - return predictions as a numpy array
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.metrics import root_mean_squared_error

TARGET_COL = "target"
FEATURE_6 = "6"
FEATURE_7 = "7"

RANDOM_STATE = 42
N_ESTIMATORS = 200
LEARNING_RATE = 0.05


def _missing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return the required columns that are absent from the dataset."""
    return [c for c in columns if c not in df.columns]


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError if any required column is missing."""
    if missing := _missing_columns(df, columns):
        raise ValueError(f"dataset is missing required columns: {missing}")


def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns to numeric; invalid values become NaN."""
    return df.apply(pd.to_numeric, errors="coerce")


def _drop_missing_rows(
    df: pd.DataFrame, subset: list[str], reason: str
) -> pd.DataFrame:
    """Drop rows with NaN in ``subset``, warning about how many were skipped."""
    clean = df.dropna(subset=subset)
    if skipped := len(df) - len(clean):
        logger.warning(f"Skipped {skipped} rows with {reason}")
    return clean


class BaseModel(ABC):
    """Common interface of all model strategies."""

    @abstractmethod
    def is_suitable(self, df: pd.DataFrame) -> bool:
        """Return True if this strategy fits the dataset."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Train the model on the dataset."""

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predictions for the dataset."""


class FormulaModel(BaseModel):
    """Predicts ``FEATURE_6 ** 2 + FEATURE_7``; nothing to train."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    @staticmethod
    def _compute(x: pd.DataFrame) -> pd.Series:
        return x[FEATURE_6] ** 2 + x[FEATURE_7]

    def is_suitable(self, df: pd.DataFrame) -> bool:
        """Check that the formula reproduces ``target`` (RMSE below threshold).

        Rows with missing or non-numeric values are skipped with a warning.
        """
        columns = [FEATURE_6, FEATURE_7, TARGET_COL]
        if missing := _missing_columns(df, columns):
            logger.warning(f"Columns missing for formula check: {missing}")
            return False

        valid = _drop_missing_rows(
            _to_numeric(df[columns]), columns, "missing/non-numeric values"
        )
        if valid.empty:
            return False

        score = root_mean_squared_error(valid[TARGET_COL], self._compute(valid))
        logger.info(f"Formula RMSE = {score:.6g} (threshold = {self.threshold:g})")
        return score < self.threshold

    def fit(self, df: pd.DataFrame) -> None:
        """Nothing to train."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the formula; rows with missing inputs give NaN (warned)."""
        columns = [FEATURE_6, FEATURE_7]
        _require_columns(df, columns)
        preds = self._compute(_to_numeric(df[columns]))
        if n_nan := int(preds.isna().sum()):
            logger.warning(f"{n_nan} predictions are NaN (missing/non-numeric inputs)")
        return preds.to_numpy()


class MLModel(BaseModel):
    """LGBMRegressor on all numeric features with median imputation.

    Only the fitted model and the per-feature training medians are stored;
    the feature list is the index of the medians.
    """

    def __init__(self) -> None:
        self.model = LGBMRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        self.medians: pd.Series | None = None

    def is_suitable(self, df: pd.DataFrame) -> bool:
        """Always suitable: this is the general fallback."""
        return True

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill missing feature values with the training medians (warned)."""
        if n_missing := int(X.isna().sum().sum()):
            logger.warning(f"Filling {n_missing} missing feature values with medians")
        return X.fillna(self.medians)

    def fit(self, df: pd.DataFrame) -> None:
        """Train on all numeric features.

        Rows without a target and non-numeric columns are skipped (with a
        warning). Missing feature values are filled with column medians.
        """
        _require_columns(df, [TARGET_COL])

        df = _drop_missing_rows(df, [TARGET_COL], "missing target")
        if df.empty:
            raise ValueError("no rows with a valid target")

        features = df.drop(columns=TARGET_COL)
        X = features.select_dtypes(include="number")
        if skipped := features.columns.difference(X.columns).tolist():
            logger.warning(f"Skipped non-numeric columns: {skipped}")
        if X.empty:
            raise ValueError("no numeric features found")

        # An all-NaN column has a NaN median; fall back to 0 so imputation always works.
        self.medians = X.median().fillna(0.0)
        self.model.fit(self._impute(X), df[TARGET_COL])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict with the trained model, filling gaps with training medians."""
        features = self.medians.index.tolist()
        _require_columns(df, features)
        return self.model.predict(self._impute(_to_numeric(df[features])))
