# Industrial Machine Failure Prediction

A complete predictive-maintenance project on the **AI4I 2020 Predictive Maintenance**
dataset (UCI): exploratory analysis, leakage-safe feature preparation, three trained
classifiers, 5-fold cross-validation, `GridSearchCV` tuning, a saved end-to-end
pipeline, and two ways to score new machines (command line and web form).

The project follows the 11 steps in the MCA PBL brief; every step is implemented in
code and written up with figures and tables in [`reports/REPORT.md`](reports/REPORT.md).

---

## Results at a glance

| Model | 5-fold CV F1 | Test precision | Test recall | Test F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.410 | 0.591 | 0.191 | 0.289 | 0.928 |
| Decision Tree | 0.818 | 0.845 | 0.721 | 0.778 | 0.887 |
| **Random Forest (selected)** | **0.836** | **0.947** | **0.794** | **0.864** | **0.978** |

Selected model: **Random Forest** (400 trees, `max_features='sqrt'`, no depth limit),
chosen on cross-validated F1 - never on the test set. On the 2,000 held-out machines:

- **54 of 68 failures caught** (recall 79.4%)
- **3 false alarms** out of 1,932 healthy machines (0.2%)
- Accuracy 99.2%, PR-AUC 0.886

Because only 3.39% of machines fail, accuracy alone is misleading: a constant
"No failure" prediction already scores 96.6%. Precision, recall and F1 for the
failure class drive every decision in this project.

---

## Project structure

```
Machine_failure/
├── app.py                      Streamlit web interface
├── main.py                     CLI entry point (eda / train / predict)
├── requirements.txt
├── data/
│   ├── ai4i2020.csv            UCI dataset (10,000 x 14)
│   ├── example_machine.json    one record for the JSON prediction example
│   └── new_machines_sample.csv six records for batch scoring
├── src/
│   ├── config.py               paths, schema, experiment settings
│   ├── data_loader.py          step 1: load, validate, drop leakage columns
│   ├── eda.py                  step 2: statistics + nine EDA figures
│   ├── preprocessing.py        step 3: feature engineering + sklearn pipeline
│   ├── evaluation.py           shared metric helpers
│   ├── train.py                steps 4-9: split, CV, tune, evaluate, select, report
│   ├── predict.py              step 10: validation and inference
│   └── plotting.py             shared figure style
├── models/
│   ├── machine_failure_model.joblib   complete fitted pipeline + metadata
│   └── model_card.json               model card
└── reports/
    ├── REPORT.md               generated project report (11 steps + figures)
    ├── figures/                17 PNG figures
    └── metrics/                CSV/JSON tables for every number in the report
```

---

## Quick start

Requires **Python 3.10+**.

```bash
# 1. install dependencies
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

# 2. run the full pipeline (EDA + training + report, about 80 seconds)
python main.py all

# 3. predict a single machine interactively
python main.py predict

# 4. or launch the web form
streamlit run app.py
```

`python main.py all` is reproducible: fixed `random_state=42`, stratified split,
and deterministic CV folds. It writes `reports/REPORT.md`, all figures, every
metric table and `models/machine_failure_model.joblib`.

### Useful commands

```bash
python main.py eda       # analysis and figures only
python main.py train     # training, tuning, evaluation and report

# score a machine from flags
python main.py predict --air 298.1 --process 308.6 --rpm 1550 \
                       --torque 42.8 --wear 0 --type L

# higher-risk example
python main.py predict --air 301.5 --process 312.0 --rpm 1280 \
                       --torque 68 --wear 210 --type L --explain

# score a JSON record (robust quoting on every shell)
python main.py predict --json-file data/example_machine.json

# batch-score a CSV and save the results
python main.py predict --csv data/new_machines_sample.csv --output predictions.csv
```

Example output:

```
==============================================================
                  MACHINE FAILURE PREDICTION
==============================================================
Model                 : Random Forest
Machine failure       : Yes
Failure probability   : 97.8%
Decision threshold    : 0.50
Risk level            : HIGH
--------------------------------------------------------------
Type                    : L
Air temperature         : 301.5 K
Process temperature     : 312.0 K
Rotational speed        : 1280 rpm
Torque                  : 68.0 Nm
Tool wear               : 210 min
--------------------------------------------------------------
Derived mechanical power : 9115 W
Derived temp difference  : 10.5 K
==============================================================
```

---

## Dataset and leakage control

Source: [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(licensed CC BY 4.0).

| Column | Role |
|---|---|
| `UDI`, `Product ID` | row identifiers - **removed** |
| `Type` | machine quality (L/M/H) - one-hot encoded |
| `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]` | sensor inputs |
| `Machine failure` | target (339 / 10,000 = 3.39%) |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | failure-mode flags - **removed** |

The five failure-mode flags describe *why* a machine failed, so they are
components of the target. Keeping them would give a meaningless ~100% accuracy;
they are used only for EDA storytelling and are excluded from `X`.

Two physically motivated features are derived **from inputs only**, inside the
saved pipeline:

- `Temperature difference [K] = process - air`
- `Mechanical power [W] = torque x 2*pi*rpm / 60`

---

## Modelling approach

1. **Split** - stratified 80/20 (`random_state=42`): 8,000 train / 2,000 test,
   preserving the 3.39% failure rate.
2. **Models** - Logistic Regression, Decision Tree, Random Forest, each packaged
   as `Pipeline(preprocessing -> classifier)` with class-imbalance handling.
3. **Cross-validation** - stratified 5-fold on the training set only; F1, recall,
   precision, accuracy, ROC-AUC and PR-AUC reported per fold.
4. **Tuning** - `GridSearchCV` (10 / 96 / 36 candidate combinations) optimising
   **F1 of the failure class**, which is the right target under 3.4% prevalence.
5. **Selection** - highest mean CV F1, tie-broken by CV recall. The test set is
   used only to report the final numbers.
6. **Operating point** - out-of-fold predictions show the F1-optimal threshold is
   0.45; the deployed threshold stays at 0.50 (test F1 0.864 vs 0.857) so
   probabilities remain directly interpretable. See
   `reports/figures/15_threshold_analysis.png`.

### What drives the model

The strongest signals are torque, rotational speed, mechanical power, tool wear
and the temperature difference. `Type` contributes little. This matches the
physics of the dataset: failures happen when a machine is pushed to the edge of
its torque/speed envelope or when a worn tool is used under load.

---

## Deliverables

| Deliverable | Location |
|---|---|
| Cleaned, analysed dataset | `data/ai4i2020.csv`, `reports/metrics/eda_*.csv` |
| Three trained classifiers | `src/train.py`, metrics in `reports/metrics/` |
| Model comparison table | `reports/REPORT.md`, `reports/metrics/test_tuned_metrics.csv` |
| CV results and tuned hyperparameters | `reports/metrics/cv_baseline_*.csv`, `tuning_results.csv` |
| Confusion matrices | `reports/figures/12_*.png`, `17_final_confusion_matrix.png` |
| Saved model + preprocessing pipeline | `models/machine_failure_model.joblib` |
| New-data prediction interface | `python main.py predict`, `streamlit run app.py` |
| Project report | `reports/REPORT.md` |

The `.joblib` artifact is a single dictionary containing the fitted pipeline, the
selected model name, hyperparameters, input defaults/ranges, decision threshold
and final test metrics. Scoring new data never retrains anything.

Streamlit app features: scenario presets, slider inputs bounded by the training
range, probability gauge, input-review table, model card in the sidebar, and
optional batch CSV scoring with download.

---

## Reproducibility

- `random_state = 42` for the split, CV folds, models and permutation importance.
- Versions used: Python 3.11.15, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.4.6.
- Full pipeline runtime: ~80 s on a laptop CPU.
- Every figure and table in the report is regenerated by `python main.py train`.

## Limitations

- AI4I is **synthetic**; failure modes follow threshold rules plus random
  failures. Real plant telemetry is noisier and needs revalidation.
- Only 339 positive cases exist, so metric estimates carry real uncertainty; the
  CV standard deviations in the report should be read alongside the point values.
- Random failures (`RNF`) are unpredictable by construction and cap recall.
- The model estimates *failure risk from current operating conditions*, not
  remaining useful life or time-to-failure.
- This is an educational decision-support demo, not a certified maintenance
  system.
