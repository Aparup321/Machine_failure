"""Step 3 - feature selection, feature engineering and preprocessing pipeline.

The preprocessing object is a scikit-learn transformer so that it can be
serialised together with the classifier. That guarantees new, unseen records
go through exactly the same transformations as the training data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add physically motivated features derived from the sensor readings.

    * Temperature difference [K] = process temperature - air temperature
      (how hard the machine has to work to hold its process temperature)
    * Mechanical power [W]      = torque [Nm] * angular velocity [rad/s]
      (the classic AI4I "power" signal: torque and rpm trade off)

    Both are computed from input columns only, so no target information leaks.
    """

    def __init__(self, add_engineered: bool = True):
        self.add_engineered = add_engineered

    def fit(self, X, y=None):
        self._validate(X)
        return self

    def transform(self, X):
        self._validate(X)
        X = pd.DataFrame(X).copy()

        if self.add_engineered:
            X["Temperature difference [K]"] = (
                X["Process temperature [K]"] - X["Air temperature [K]"]
            )
            X["Mechanical power [W]"] = (
                X["Torque [Nm]"] * 2.0 * np.pi * X["Rotational speed [rpm]"] / 60.0
            )

        return X

    @staticmethod
    def _validate(X) -> None:
        missing = [c for c in config.FEATURES if c not in pd.DataFrame(X).columns]
        if missing:
            raise ValueError(f"Input data is missing required columns: {missing}")

    def get_feature_names_out(self, input_features=None):
        names = list(config.FEATURES)
        if self.add_engineered:
            names += config.ENGINEERED_FEATURES
        return np.asarray(names, dtype=object)


# --------------------------------------------------------------------------
# Preprocessing pipeline
# --------------------------------------------------------------------------
def build_preprocessor(add_engineered: bool = config.USE_ENGINEERED_FEATURES) -> Pipeline:
    """FeatureEngineer -> column-wise impute/encode/scale."""
    numeric_columns = list(config.NUMERIC_FEATURES)
    if add_engineered:
        numeric_columns += config.ENGINEERED_FEATURES

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer(add_engineered=add_engineered)),
            (
                "columns",
                ColumnTransformer(
                    transformers=[
                        ("num", numeric_pipe, numeric_columns),
                        ("cat", categorical_pipe, list(config.CATEGORICAL_FEATURES)),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
        ]
    )


def build_model_pipeline(estimator, add_engineered: bool = config.USE_ENGINEERED_FEATURES) -> Pipeline:
    """Preprocessing + classifier as one deployable object."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(add_engineered)),
            ("classifier", estimator),
        ]
    )


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select the six raw inputs; the target is the only output."""
    X = df[list(config.FEATURES)].copy()
    y = df[config.TARGET].astype(int).copy()
    return X, y
