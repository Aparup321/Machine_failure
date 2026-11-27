"""Central configuration for the Industrial Machine Failure Prediction project."""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

RAW_DATA_PATH = DATA_DIR / "ai4i2020.csv"
MODEL_PATH = MODELS_DIR / "machine_failure_model.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"

for _folder in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, METRICS_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset schema
# --------------------------------------------------------------------------
TARGET = "Machine failure"

# Row identifiers - no predictive meaning, must not enter the model.
ID_COLUMNS = ["UDI", "Product ID"]

# Failure-mode flags. These are *derived from* / components of the target,
# so they would leak the answer. They are used only for EDA storytelling.
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Physically motivated features derived ONLY from the sensor inputs above.
ENGINEERED_FEATURES = ["Temperature difference [K]", "Mechanical power [W]"]
USE_ENGINEERED_FEATURES = True

TYPE_CATEGORIES = ["L", "M", "H"]

# --------------------------------------------------------------------------
# Experiment settings
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SCORING = "f1"          # positive (failure) class is what matters
THRESHOLD = 0.5

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest"]

# Human-readable labels for the failure class
LABELS = {0: "No", 1: "Yes"}
