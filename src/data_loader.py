"""Step 1 - loading and inspecting the AI4I 2020 predictive maintenance dataset."""

from __future__ import annotations

import pandas as pd

from . import config


def load_raw_data(path=None) -> pd.DataFrame:
    """Read the raw CSV exactly as distributed by UCI."""
    path = path or config.RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download it from:\n"
            "  https://archive.ics.uci.edu/static/public/601/"
            "ai4i+2020+predictive+maintenance+dataset.zip\n"
            "and place ai4i2020.csv inside the data/ folder."
        )
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    """Fail loudly if the CSV is not the dataset this project expects."""
    required = (
        config.ID_COLUMNS
        + config.FAILURE_MODE_COLUMNS
        + config.FEATURES
        + [config.TARGET]
    )
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    if df[config.TARGET].isin([0, 1]).all() is False:
        raise ValueError("Target column must be binary (0/1).")


def load_clean_data(path=None) -> pd.DataFrame:
    """Load, validate and drop identifier / leakage columns.

    Returns a dataframe with the six raw sensor inputs plus the target.
    Nothing target-derived survives into X.
    """
    df = load_raw_data(path)
    validate_schema(df)
    return df.drop(columns=config.ID_COLUMNS + config.FAILURE_MODE_COLUMNS)


def describe_dataset(df: pd.DataFrame) -> dict:
    """Compact structural summary used by the report."""
    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "n_failures": int(df[config.TARGET].sum()),
        "n_healthy": int((df[config.TARGET] == 0).sum()),
        "failure_rate_pct": round(100 * df[config.TARGET].mean(), 2),
        "type_counts": {k: int(v) for k, v in df["Type"].value_counts().items()},
        "failure_modes": {
            column: int(df[column].sum())
            for column in config.FAILURE_MODE_COLUMNS
            if column in df.columns
        },
    }
    return summary


def print_dataset_overview(df: pd.DataFrame) -> None:
    """Human-friendly console overview (step 1 expected output)."""
    print("=" * 74)
    print("DATASET OVERVIEW".center(74))
    print("=" * 74)
    print(f"Shape            : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Missing values   : {int(df.isna().sum().sum())}")
    print(f"Duplicate rows   : {int(df.duplicated().sum())}")
    print(f"Failures         : {int(df[config.TARGET].sum())} "
          f"({100 * df[config.TARGET].mean():.2f}%)")
    print(f"Machine types    : {df['Type'].value_counts().to_dict()}")
    print("\nColumns / dtypes")
    print("-" * 74)
    for column, dtype in df.dtypes.items():
        print(f"  {column:<28} {str(dtype):<10}")
    print("\nFirst five rows")
    print("-" * 74)
    print(df.head().to_string(index=False))
    print()
