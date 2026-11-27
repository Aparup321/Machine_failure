"""Industrial Machine Failure Prediction - command line entry point.

Usage examples
--------------
  python main.py all                 # EDA + training + saved model (first run)
  python main.py eda                 # EDA only
  python main.py train               # training / tuning / evaluation only
  python main.py predict             # interactive prediction prompt
  python main.py predict --air 298.1 --process 308.6 --rpm 1550 \
                         --torque 42.8 --wear 0 --type L --explain
  python main.py predict --csv data/new_machines.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Industrial Machine Failure Prediction System",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("eda", help="run exploratory data analysis and write figures")
    sub.add_parser("train", help="train, cross-validate, tune, evaluate and save the model")
    sub.add_parser("all", help="run EDA and training end to end")

    predict = sub.add_parser("predict", help="predict failure for new machine data")
    predict.add_argument("--air", type=float, help="air temperature in Kelvin")
    predict.add_argument("--process", type=float, help="process temperature in Kelvin")
    predict.add_argument("--rpm", type=float, help="rotational speed in rpm")
    predict.add_argument("--torque", type=float, help="torque in Nm")
    predict.add_argument("--wear", type=float, help="tool wear in minutes")
    predict.add_argument("--type", dest="machine_type", default="L",
                         choices=config.TYPE_CATEGORIES, help="machine quality (default: L)")
    predict.add_argument("--threshold", type=float, default=None,
                         help="decision threshold (default: stored model threshold)")
    predict.add_argument("--json", type=str, default=None,
                         help="score a JSON record instead of flags")
    predict.add_argument("--json-file", type=str, default=None,
                         help="path to a JSON file with one record")
    predict.add_argument("--csv", type=str, default=None,
                         help="batch-score a CSV file")
    predict.add_argument("--output", type=str, default=None,
                         help="output path for --csv scoring")
    predict.add_argument("--model", type=str, default=None,
                         help="path to a different .joblib model")
    predict.add_argument("--explain", action="store_true",
                         help="print a short explanation of the drivers")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "all", "eda"):
        from src.eda import run_eda

        run_eda()

    if args.command in (None, "all", "train"):
        from src.train import run_training

        run_training()

    if args.command == "predict":
        from src.predict import run_prediction

        try:
            return run_prediction(args)
        except (FileNotFoundError, ValueError) as error:
            print(f"Error: {error}")
            return 1

    if args.command is None:
        print("\nNext steps:")
        print("  python main.py predict      # interactive prediction")
        print("  streamlit run app.py        # web interface")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
