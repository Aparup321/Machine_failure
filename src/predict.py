"""Step 10 - score new, unseen machine records with the saved pipeline.

All inference goes through `models/machine_failure_model.joblib`, which contains
the fitted preprocessing + classifier, so no retraining is ever required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # allow running this file from anywhere
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_bundle(path: Path | str | None = None) -> dict:
    import joblib

    path = Path(path) if path else config.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run:  python main.py train"
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("Unexpected model artifact format.")
    return bundle


# ----------------------------------------------------------------------
# Validation and inference
# ----------------------------------------------------------------------
def build_record(
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
    machine_type: str = "L",
) -> dict:
    machine_type = str(machine_type).upper()
    if machine_type not in config.TYPE_CATEGORIES:
        raise ValueError(f"Type must be one of {config.TYPE_CATEGORIES}, got {machine_type!r}")

    record = {
        "Type": machine_type,
        "Air temperature [K]": float(air_temperature),
        "Process temperature [K]": float(process_temperature),
        "Rotational speed [rpm]": float(rotational_speed),
        "Torque [Nm]": float(torque),
        "Tool wear [min]": float(tool_wear),
    }
    _sanity_check(record)
    return record


def _sanity_check(record: dict) -> None:
    """Reject physically impossible values before they reach the model."""
    checks = {
        "Air temperature [K]": (250, 350),
        "Process temperature [K]": (250, 400),
        "Rotational speed [rpm]": (0, 10000),
        "Torque [Nm]": (0, 500),
        "Tool wear [min]": (0, 10000),
    }
    for column, (low, high) in checks.items():
        value = record[column]
        if not low <= value <= high:
            raise ValueError(
                f"{column} = {value} is outside the plausible range [{low}, {high}]."
            )
    if record["Process temperature [K]"] < record["Air temperature [K]"]:
        raise ValueError(
            "Process temperature cannot be below air temperature "
            f"({record['Process temperature [K]']} < {record['Air temperature [K]']})."
        )


def _range_warnings(record: dict, bundle: dict) -> list[str]:
    warnings = []
    for column, (low, high) in bundle.get("input_ranges", {}).items():
        if column in record and not low <= record[column] <= high:
            warnings.append(
                f"{column} = {record[column]:g} is outside the training range "
                f"[{low:g}, {high:g}]; the prediction is an extrapolation."
            )
    return warnings


def predict_record(bundle: dict, record: dict, threshold: float | None = None) -> dict:
    """Predict a single machine. Returns label, probability and context."""
    threshold = bundle.get("threshold", config.THRESHOLD) if threshold is None else threshold
    frame = pd.DataFrame([record])[list(bundle["features"])]

    pipeline = bundle["pipeline"]
    probability = float(pipeline.predict_proba(frame)[0, 1])
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "label": config.LABELS[prediction],
        "probability": probability,
        "probability_pct": round(100 * probability, 2),
        "threshold": float(threshold),
        "risk_level": risk_level(probability),
        "model_name": bundle.get("model_name", "unknown"),
        "warnings": _range_warnings(record, bundle),
        "record": record,
        "engineered": {
            "Mechanical power [W]": round(
                record["Torque [Nm]"] * 2 * 3.141592653589793
                * record["Rotational speed [rpm]"] / 60, 1
            ),
            "Temperature difference [K]": round(
                record["Process temperature [K]"] - record["Air temperature [K]"], 1
            ),
        },
    }


def predict_dataframe(bundle: dict, df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Batch scoring for a dataframe with the six raw input columns."""
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing columns: {missing}")

    threshold = bundle.get("threshold", config.THRESHOLD) if threshold is None else threshold
    probas = bundle["pipeline"].predict_proba(df[list(bundle["features"])])[:, 1]
    out = df.copy()
    out["failure_probability"] = probas.round(4)
    out["prediction"] = (probas >= threshold).astype(int)
    out["prediction_label"] = [config.LABELS[int(p)] for p in out["prediction"]]
    out["risk_level"] = [risk_level(p) for p in probas]
    return out


def risk_level(probability: float) -> str:
    if probability >= 0.60:
        return "HIGH"
    if probability >= 0.30:
        return "MODERATE"
    return "LOW"


# ----------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------
def format_result(result: dict, explain: bool = False) -> str:
    probability = result["probability"]
    probability_text = (
        "<0.01%" if probability < 0.0001
        else f"{probability:.2%}" if probability < 0.01
        else f"{probability:.1%}"
    )
    lines = [
        "",
        "=" * 62,
        "MACHINE FAILURE PREDICTION".center(62),
        "=" * 62,
        f"Model                 : {result['model_name']}",
        f"Machine failure       : {result['label']}",
        f"Failure probability   : {probability_text}",
        f"Decision threshold    : {result['threshold']:.2f}",
        f"Risk level            : {result['risk_level']}",
        "-" * 62,
        f"{'Type':<24}: {result['record']['Type']}",
        f"{'Air temperature':<24}: {result['record']['Air temperature [K]']:.1f} K",
        f"{'Process temperature':<24}: {result['record']['Process temperature [K]']:.1f} K",
        f"{'Rotational speed':<24}: {result['record']['Rotational speed [rpm]']:.0f} rpm",
        f"{'Torque':<24}: {result['record']['Torque [Nm]']:.1f} Nm",
        f"{'Tool wear':<24}: {result['record']['Tool wear [min]']:.0f} min",
        "-" * 62,
        f"Derived mechanical power : {result['engineered']['Mechanical power [W]']:.0f} W",
        f"Derived temp difference  : {result['engineered']['Temperature difference [K]']:.1f} K",
    ]
    if explain:
        lines.append("-" * 62)
        lines.append("Note: in this dataset the strongest drivers are")
        lines.append("mechanical power, tool wear and torque. High power or high")
        lines.append("wear combined with high torque pushes risk up.")
    for warning in result["warnings"]:
        lines.append(f"WARNING: {warning}")
    lines.append("=" * 62)
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Interactive CLI
# ----------------------------------------------------------------------
def _ask(prompt: str, default, cast, low=None, high=None):
    while True:
        suffix = f" [{default}]"
        if low is not None and high is not None:
            suffix += f"  (range {low:g}-{high:g})"
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw == "":
            return default
        try:
            value = cast(raw)
        except ValueError:
            print("  Please enter a valid number.")
            continue
        if low is not None and not low <= value <= high:
            print(f"  Value must be between {low:g} and {high:g}.")
            continue
        return value


def interactive_cli(bundle: dict, explain: bool = False) -> dict:
    defaults = bundle.get("input_defaults", {})
    ranges = bundle.get("input_ranges", {})

    def bounds(column, fallback):
        if column in ranges:
            low, high = ranges[column]
            pad = (high - low) * 0.25
            return round(low - pad, 1), round(high + pad, 1)
        return fallback

    print("\nEnter the machine measurements (press Enter to accept the default).\n")
    machine_type = input(
        f"Type L / M / H [{defaults.get('Type', 'L')}]: "
    ).strip().upper() or defaults.get("Type", "L")
    while machine_type not in config.TYPE_CATEGORIES:
        machine_type = input("  Type must be L, M or H: ").strip().upper()

    air = _ask("Air temperature [K]", defaults.get("Air temperature [K]", 298.1),
               float, *bounds("Air temperature [K]", (250, 350)))
    process = _ask("Process temperature [K]", defaults.get("Process temperature [K]", 308.6),
                   float, *bounds("Process temperature [K]", (250, 400)))
    if process < air:
        print("  Process temperature was below air temperature; swapped them.")
        air, process = process, air
    rpm = _ask("Rotational speed [rpm]", defaults.get("Rotational speed [rpm]", 1500),
               float, *bounds("Rotational speed [rpm]", (0, 5000)))
    torque = _ask("Torque [Nm]", defaults.get("Torque [Nm]", 40.0),
                  float, *bounds("Torque [Nm]", (0, 200)))
    wear = _ask("Tool wear [min]", defaults.get("Tool wear [min]", 0),
                float, *bounds("Tool wear [min]", (0, 300)))

    record = build_record(air, process, rpm, torque, wear, machine_type)
    result = predict_record(bundle, record)
    print(format_result(result, explain=explain))
    return result


# ----------------------------------------------------------------------
# Entry point used by main.py
# ----------------------------------------------------------------------
def run_prediction(args) -> int:
    bundle = load_bundle(getattr(args, "model", None))

    if getattr(args, "csv", None):
        frame = pd.read_csv(args.csv)
        scored = predict_dataframe(bundle, frame, getattr(args, "threshold", None))
        destination = args.output or "predictions.csv"
        scored.to_csv(destination, index=False)
        print(f"Scored {len(scored)} rows -> {destination}")
        print(f"Predicted failures: {int(scored['prediction'].sum())} "
              f"({100 * scored['prediction'].mean():.1f}%)")
        return 0

    if getattr(args, "json", None) or getattr(args, "json_file", None):
        if getattr(args, "json_file", None):
            payload = Path(args.json_file).read_text(encoding="utf-8")
        else:
            payload = args.json
        record = json.loads(payload)
        required = ["Air temperature [K]", "Process temperature [K]",
                    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
        missing = [c for c in required if c not in record]
        if missing:
            print(f"JSON record is missing: {missing}")
            return 1
        record = build_record(
            record["Air temperature [K]"], record["Process temperature [K]"],
            record["Rotational speed [rpm]"], record["Torque [Nm]"],
            record["Tool wear [min]"], record.get("Type", "L"),
        )
        result = predict_record(bundle, record, getattr(args, "threshold", None))
        print(format_result(result, explain=getattr(args, "explain", False)))
        return 0

    provided = [getattr(args, "air", None), getattr(args, "process", None),
                getattr(args, "rpm", None), getattr(args, "torque", None),
                getattr(args, "wear", None)]
    if all(value is None for value in provided):
        interactive_cli(bundle, explain=getattr(args, "explain", False))
        return 0

    missing = [
        name for name, value in zip(
            ["--air", "--process", "--rpm", "--torque", "--wear"], provided
        ) if value is None
    ]
    if missing:
        print(f"Missing arguments: {', '.join(missing)}")
        print("Run without arguments for the interactive prompt.")
        return 1

    record = build_record(
        args.air, args.process, args.rpm, args.torque, args.wear,
        getattr(args, "machine_type", "L") or "L",
    )
    result = predict_record(bundle, record, getattr(args, "threshold", None))
    print(format_result(result, explain=getattr(args, "explain", False)))
    return 0
