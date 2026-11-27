"""Steps 4-9 - split, train, evaluate, cross-validate, tune and select.

Run with:  python main.py train

Artifacts
---------
models/machine_failure_model.joblib   deployable pipeline + metadata
models/model_card.json                human readable model card
reports/metrics/*.csv|json            every number used in the report
reports/figures/*.png                 comparison, CV, confusion, curves
reports/REPORT.md                     generated project report
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier

from . import config
from .data_loader import load_clean_data
from .evaluation import (
    METRIC_COLUMNS,
    PRETTY,
    compute_metrics,
    markdown_table,
    metrics_row,
)
from .plotting import MODEL_COLORS, PALETTE, save_figure
from .preprocessing import build_model_pipeline, split_features_target

# zero_division=0 keeps folds that predict no failure from raising warnings
# while still scoring those folds as 0 instead of silently dropping them.
CV_SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

TUNING_SCORING = {
    "f1": make_scorer(f1_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "accuracy": "accuracy",
}


# ----------------------------------------------------------------------
# Model definitions and search spaces
# ----------------------------------------------------------------------
def build_baselines() -> dict:
    """Default, untuned estimators (sensible class-imbalance handling)."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, solver="liblinear", class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=config.RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
    }


def build_search_spaces() -> dict:
    """GridSearchCV settings for each model (step 8)."""
    spaces = {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=3000, solver="liblinear", random_state=config.RANDOM_STATE,
            ),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=config.RANDOM_STATE),
            "param_grid": {
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [3, 5, 7, 10, 15, None],
                "classifier__min_samples_leaf": [1, 5, 20, 50],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                random_state=config.RANDOM_STATE, n_jobs=1,
            ),
            "param_grid": {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [8, 12, None],
                "classifier__min_samples_leaf": [1, 3, 5],
                "classifier__max_features": ["sqrt"],
                "classifier__class_weight": [None, "balanced_subsample"],
            },
        },
    }
    return spaces


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def prepare_data():
    df = load_clean_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    return X, y, X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
# Cross-validation (step 7)
# ----------------------------------------------------------------------
def cross_validate_baselines(models: dict, X_train, y_train, cv) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    for name, estimator in models.items():
        pipeline = build_model_pipeline(estimator)
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=CV_SCORING,
            n_jobs=-1, error_score="raise",
        )
        for fold in range(cv.get_n_splits()):
            row = {"model": name, "fold": fold + 1}
            row.update({m: scores[f"test_{m}"][fold] for m in CV_SCORING})
            fold_rows.append(row)

    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby("model")[list(CV_SCORING)]
        .agg(["mean", "std"])
        .reindex(list(models))
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.round(4)
    return folds, summary


# ----------------------------------------------------------------------
# Baseline test evaluation (step 6)
# ----------------------------------------------------------------------
def evaluate_baselines(models: dict, X_train, y_train, X_test, y_test):
    rows, fitted = [], {}
    for name, estimator in models.items():
        pipeline = build_model_pipeline(estimator).fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, (proba >= config.THRESHOLD).astype(int), proba)
        rows.append(metrics_row(name, metrics))
        fitted[name] = pipeline
    return pd.DataFrame(rows), fitted


# ----------------------------------------------------------------------
# Hyperparameter tuning (step 8)
# ----------------------------------------------------------------------
def tune_models(spaces: dict, X_train, y_train, cv) -> tuple[dict, pd.DataFrame]:
    searches, rows = {}, []
    for name, spec in spaces.items():
        pipeline = build_model_pipeline(spec["estimator"])
        search = GridSearchCV(
            pipeline,
            param_grid=spec["param_grid"],
            scoring=TUNING_SCORING,
            refit="f1",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )
        start = time.perf_counter()
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        searches[name] = search

        results = search.cv_results_
        best_index = int(np.argmax(results["mean_test_f1"]))
        rows.append(
            {
                "model": name,
                "best_cv_f1": results["mean_test_f1"][best_index],
                "best_cv_f1_std": results["std_test_f1"][best_index],
                "best_cv_recall": results["mean_test_recall"][best_index],
                "best_cv_precision": results["mean_test_precision"][best_index],
                "best_cv_accuracy": results["mean_test_accuracy"][best_index],
                "candidates_tested": len(results["params"]),
                "fits": len(results["params"]) * cv.get_n_splits(),
                "search_seconds": round(elapsed, 2),
                "best_params": json.dumps(_clean_params(search.best_params_)),
            }
        )

        # Full grid, for an appendix and the tuning plot.
        grid = pd.DataFrame(
            {
                "model": name,
                "params": [json.dumps(_clean_params(p)) for p in results["params"]],
                "mean_f1": results["mean_test_f1"],
                "std_f1": results["std_test_f1"],
                "mean_recall": results["mean_test_recall"],
                "mean_precision": results["mean_test_precision"],
            }
        ).sort_values("mean_f1", ascending=False)
        grid.to_csv(
            config.METRICS_DIR / f"grid_{name.lower().replace(' ', '_')}.csv",
            index=False,
        )

    return searches, pd.DataFrame(rows)


def _clean_params(params: dict) -> dict:
    return {key.replace("classifier__", ""): value for key, value in params.items()}


# ----------------------------------------------------------------------
# Tuned evaluation (step 9)
# ----------------------------------------------------------------------
def evaluate_tuned(searches: dict, X_test, y_test):
    rows, probas = [], {}
    for name, search in searches.items():
        pipeline = search.best_estimator_
        proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, (proba >= config.THRESHOLD).astype(int), proba)
        rows.append(metrics_row(name, metrics))
        probas[name] = proba
    return pd.DataFrame(rows), probas


def select_best(tuning_table: pd.DataFrame) -> str:
    """Select on cross-validated F1, breaking ties with recall."""
    ranked = tuning_table.sort_values(
        ["best_cv_f1", "best_cv_recall"], ascending=False
    )
    return str(ranked.iloc[0]["model"])


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def plot_cv_scores(folds: pd.DataFrame, path_name="10_cv_f1_scores.png"):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    order = list(dict.fromkeys(folds["model"]))
    data = [folds.loc[folds["model"] == m, "f1"].values for m in order]
    parts = ax.boxplot(data, patch_artist=True, widths=0.45, showmeans=True,
                       meanprops={"marker": "D", "markerfacecolor": "white",
                                  "markeredgecolor": "black", "markersize": 5})
    for patch, colour in zip(parts["boxes"], MODEL_COLORS):
        patch.set_facecolor(colour)
        patch.set_alpha(0.65)
    for i, (values, colour) in enumerate(zip(data, MODEL_COLORS), start=1):
        jitter = np.random.default_rng(config.RANDOM_STATE + i).normal(0, 0.045, len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=28,
                   color=colour, edgecolor="black", linewidth=0.4, zorder=3)

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order)
    ax.set_ylabel("5-fold CV F1-score (failure class)")
    ax.set_title("Cross-validation consistency on the training set")
    for i, values in enumerate(data, start=1):
        ax.text(i, max(values) + 0.012, f"mean {np.mean(values):.3f}",
                ha="center", fontsize=8.5)
    ax.set_ylim(min(min(d) for d in data) - 0.06, max(max(d) for d in data) + 0.05)
    fig.tight_layout()
    save_figure(fig, path_name)


def plot_cv_metrics(summary: pd.DataFrame, path_name="11_cv_metrics_comparison.png"):
    metrics = ["f1", "recall", "precision", "accuracy"]
    models = list(summary.index)
    x = np.arange(len(models))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for offset, metric in zip(np.linspace(-1.5, 1.5, len(metrics)), metrics):
        values = summary[f"{metric}_mean"].values
        errors = summary[f"{metric}_std"].values
        ax.bar(x + offset * width, values, width, yerr=errors, capsize=3,
               label=PRETTY[metric], color=MODEL_COLORS[metrics.index(metric)],
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean 5-fold CV score")
    ax.set_title("Metric comparison before tuning (mean +/- std across folds)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save_figure(fig, path_name)


def plot_confusion_matrices(probas_or_preds, y_test, title, filename):
    """3-panel confusion matrices for a dict {model: y_pred}."""
    actual = np.asarray(y_test)
    fig, axes = plt.subplots(1, len(probas_or_preds), figsize=(4.3 * len(probas_or_preds), 4))
    axes = np.atleast_1d(axes)
    for ax, (name, y_pred) in zip(axes, probas_or_preds.items()):
        # Build the matrix from raw arrays: pandas would align the Series by
        # index and silently drop rows whose original index > n_test.
        predicted = np.asarray(y_pred)
        cm = pd.crosstab(
            pd.Series(actual, name="Actual"),
            pd.Series(predicted, name="Predicted"),
        ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        ax.imshow(cm.values, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                value = int(cm.values[i, j])
                share = 100 * value / cm.values[i].sum() if cm.values[i].sum() else 0
                ax.text(j, i, f"{value}\n({share:.1f}%)", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if value > cm.values.max() * 0.55 else "#1d3557")
        ax.set_xticks([0, 1], labels=["No", "Yes"])
        ax.set_yticks([0, 1], labels=["No", "Yes"])
        ax.set_xlabel("Predicted failure")
        ax.set_ylabel("Actual failure")
        ax.set_title(name, fontsize=10.5)
        ax.grid(False)
    fig.suptitle(title, fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, filename)


def plot_final_confusion(y_test, y_pred, metrics: dict, model_name: str) -> None:
    """Large, presentation-ready confusion matrix for the selected model."""
    actual = np.asarray(y_test)
    predicted = np.asarray(y_pred)
    cm = pd.crosstab(
        pd.Series(actual, name="Actual"), pd.Series(predicted, name="Predicted")
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(cm.values, cmap="Blues", vmin=0, vmax=cm.values.max() * 1.15)
    labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
    for i in range(2):
        for j in range(2):
            value = int(cm.values[i, j])
            share = 100 * value / cm.values[i].sum() if cm.values[i].sum() else 0
            ax.text(j, i, f"{labels[i][j]}\n{value}\n({share:.1f}% of row)",
                    ha="center", va="center", fontsize=10.5,
                    color="white" if value > cm.values.max() * 0.55 else "#1d3557")
    ax.set_xticks([0, 1], labels=["No failure", "Failure"])
    ax.set_yticks([0, 1], labels=["No failure", "Failure"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"{model_name} - test set\n"
        f"accuracy {metrics['accuracy']:.3f} | precision {metrics['precision']:.3f} | "
        f"recall {metrics['recall']:.3f} | F1 {metrics['f1']:.3f}",
        fontsize=10.5,
    )
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, "17_final_confusion_matrix.png")


def plot_roc_pr(probas: dict, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for colour, (name, proba) in zip(MODEL_COLORS, probas.items()):
        fpr, tpr, _ = roc_curve(y_test, proba)
        axes[0].plot(fpr, tpr, color=colour, linewidth=1.8,
                     label=f"{name} (AUC {compute_metrics(y_test, (proba >= .5).astype(int), proba)['roc_auc']:.3f})")
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = compute_metrics(y_test, (proba >= .5).astype(int), proba)["pr_auc"]
        axes[1].plot(recall, precision, color=colour, linewidth=1.8,
                     label=f"{name} (AP {ap:.3f})")

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curves - tuned models (test set)")
    axes[0].legend(loc="lower right", fontsize=8)

    baseline = y_test.mean()
    axes[1].axhline(baseline, linestyle="--", color="gray", linewidth=1,
                    label=f"Chance (prevalence {baseline:.3f})")
    axes[1].set_xlabel("Recall (failure class)")
    axes[1].set_ylabel("Precision (failure class)")
    axes[1].set_title("Precision-Recall curves - tuned models (test set)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "13_roc_pr_curves.png")


def plot_importance(pipeline, X_test, y_test, model_name):
    """Permutation importance on the raw inputs + the model's own ranking.

    Permutation is applied to the six columns the user actually supplies
    (the deployed pipeline performs the engineering internally), while the
    built-in importances are already expressed on the transformed features.
    """
    raw_names = list(config.FEATURES)
    engineered_names = list(
        pipeline.named_steps["preprocessor"].named_steps["columns"]
        .get_feature_names_out()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    result = permutation_importance(
        pipeline, X_test, y_test, scoring="f1", n_repeats=10,
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    order = np.argsort(result.importances_mean)
    axes[0].barh(np.array(raw_names)[order], result.importances_mean[order],
                 xerr=result.importances_std[order], color="#457B9D", capsize=3)
    axes[0].set_xlabel("Drop in F1 when the raw feature is shuffled")
    axes[0].set_title(f"Permutation importance - {model_name} (test set)")
    axes[0].tick_params(labelsize=8)

    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        title = f"Built-in impurity importance - {model_name}"
    else:
        values = np.abs(classifier.coef_[0])
        title = f"Absolute coefficients - {model_name}"
    order = np.argsort(values)
    axes[1].barh(np.array(engineered_names)[order], values[order], color="#E76F51")
    axes[1].set_xlabel("Relative importance")
    axes[1].set_title(title)
    axes[1].tick_params(labelsize=8)

    fig.tight_layout()
    save_figure(fig, "14_feature_importance.png")

    # Two tables: raw features (comparable across models) and model-internal.
    permutation_table = pd.DataFrame(
        {
            "feature": raw_names,
            "permutation_importance": result.importances_mean,
            "permutation_std": result.importances_std,
        }
    ).sort_values("permutation_importance", ascending=False)
    permutation_table.round(4).to_csv(
        config.METRICS_DIR / "permutation_importance.csv", index=False
    )

    model_table = pd.DataFrame(
        {"transformed_feature": engineered_names, "model_importance": values}
    ).sort_values("model_importance", ascending=False)
    model_table.round(4).to_csv(
        config.METRICS_DIR / "model_feature_importance.csv", index=False
    )

    return permutation_table



def plot_threshold_analysis(y_train, oof_proba):
    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for threshold in thresholds:
        pred = (oof_proba >= threshold).astype(int)
        metrics = compute_metrics(y_train, pred)
        rows.append({"threshold": threshold, **metrics})
    table = pd.DataFrame(rows)
    table.to_csv(config.METRICS_DIR / "threshold_analysis_oof.csv", index=False)

    best = table.loc[table["f1"].idxmax()]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(table["threshold"], table["precision"], label="Precision", color="#2A9D8F")
    ax.plot(table["threshold"], table["recall"], label="Recall", color="#E76F51")
    ax.plot(table["threshold"], table["f1"], label="F1-score", color="#264653", linewidth=2.2)
    ax.axvline(config.THRESHOLD, linestyle="--", color="gray", linewidth=1,
               label=f"Deployed threshold = {config.THRESHOLD:.2f}")
    ax.scatter([best["threshold"]], [best["f1"]], color="#E9C46A", s=70,
               edgecolor="black", zorder=4)
    ax.annotate(
        f"best F1 {best['f1']:.3f}\n@ t={best['threshold']:.2f}",
        xy=(best["threshold"], best["f1"]),
        xytext=(best["threshold"] + 0.06, best["f1"] - 0.12),
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 0.8},
    )
    ax.set_xlabel("Decision threshold for 'failure'")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Operating point analysis (out-of-fold predictions, training data)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save_figure(fig, "15_threshold_analysis.png")
    return table, float(best["threshold"]), float(best["f1"])


def plot_learning_curve(pipeline, X_train, y_train, cv):
    sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train, cv=cv, scoring="f1",
        train_sizes=np.linspace(0.15, 1.0, 6), n_jobs=-1, shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(sizes, train_scores.mean(axis=1), "o-", color="#2A9D8F", label="Training F1")
    ax.fill_between(sizes, train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1), alpha=0.15, color="#2A9D8F")
    ax.plot(sizes, val_scores.mean(axis=1), "o-", color="#E76F51", label="Validation F1")
    ax.fill_between(sizes, val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1), alpha=0.15, color="#E76F51")
    ax.set_xlabel("Training records used")
    ax.set_ylabel("F1-score (failure class)")
    ax.set_title("Learning curve - selected model")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    save_figure(fig, "16_learning_curve.png")
    return pd.DataFrame(
        {"training_records": sizes,
         "train_f1_mean": train_scores.mean(1),
         "validation_f1_mean": val_scores.mean(1),
         "validation_f1_std": val_scores.std(1)}
    )


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run_training(verbose: bool = True) -> dict:
    started = time.perf_counter()
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )

    X, y, X_train, X_test, y_train, y_test = prepare_data()
    if verbose:
        print("=" * 74)
        print("TRAINING PIPELINE".center(74))
        print("=" * 74)
        print(f"Full dataset      : {len(X):,} rows, {X.shape[1]} input features")
        print(f"Training set      : {len(X_train):,} rows "
              f"({y_train.sum()} failures, {100 * y_train.mean():.2f}%)")
        print(f"Test set          : {len(X_test):,} rows "
              f"({y_test.sum()} failures, {100 * y_test.mean():.2f}%)")
        print(f"Cross-validation  : Stratified {config.CV_FOLDS}-fold")
        print(f"Tuning metric     : {config.SCORING.upper()} (failure class)\n")

    # ---------------- Step 5/6: baselines -----------------------------
    baselines = build_baselines()
    baseline_folds, baseline_cv = cross_validate_baselines(baselines, X_train, y_train, cv)
    baseline_test, _ = evaluate_baselines(baselines, X_train, y_train, X_test, y_test)

    baseline_folds.round(4).to_csv(config.METRICS_DIR / "cv_baseline_folds.csv", index=False)
    baseline_cv.to_csv(config.METRICS_DIR / "cv_baseline_summary.csv")
    baseline_test.round(4).to_csv(config.METRICS_DIR / "test_baseline_metrics.csv", index=False)

    if verbose:
        print("Step 5-6 | Baseline models")
        print("-" * 74)
        print(baseline_test[["model"] + METRIC_COLUMNS].round(3).to_string(index=False))
        print("\n5-fold CV (untuned, mean):")
        print(baseline_cv[["f1_mean", "recall_mean", "precision_mean", "accuracy_mean"]]
              .round(3).to_string(), "\n")

    # ---------------- Step 8: tuning -----------------------------------
    spaces = build_search_spaces()
    searches, tuning_table = tune_models(spaces, X_train, y_train, cv)
    tuning_table.round(4).to_csv(config.METRICS_DIR / "tuning_results.csv", index=False)

    if verbose:
        print("Step 8 | GridSearchCV results")
        print("-" * 74)
        for row in tuning_table.itertuples(index=False):
            print(f"{row.model:<22} best CV F1 = {row.best_cv_f1:.4f} "
                  f"(+/- {row.best_cv_f1_std:.4f})  "
                  f"{row.candidates_tested} candidates / {row.fits} fits "
                  f"in {row.search_seconds:.1f}s")
            print(f"{'':<22} {row.best_params}")
        print()

    # ---------------- Step 9: tuned test performance -------------------
    tuned_test, tuned_probas = evaluate_tuned(searches, X_test, y_test)
    tuned_test.round(4).to_csv(config.METRICS_DIR / "test_tuned_metrics.csv", index=False)

    selected_name = select_best(tuning_table)
    selected_search = searches[selected_name]
    selected_pipeline = selected_search.best_estimator_

    if verbose:
        print("Step 9 | Tuned models on the untouched test set")
        print("-" * 74)
        print(tuned_test[["model"] + METRIC_COLUMNS].round(3).to_string(index=False))
        print(f"\n>>> Selected model: {selected_name}")
        print(f"    Chosen on CV F1 = "
              f"{tuning_table.set_index('model').loc[selected_name, 'best_cv_f1']:.4f}\n")

    # ---------------- Threshold analysis (out-of-fold) -----------------
    oof_proba = cross_val_predict(
        selected_pipeline, X_train, y_train, cv=cv,
        method="predict_proba", n_jobs=-1,
    )[:, 1]
    threshold_table, best_threshold, best_threshold_f1 = plot_threshold_analysis(
        y_train, oof_proba
    )

    # Final selected-model evaluation at the deployed threshold (0.5)
    selected_proba = tuned_probas[selected_name]
    selected_pred = (selected_proba >= config.THRESHOLD).astype(int)
    final_metrics = compute_metrics(y_test, selected_pred, selected_proba)

    alt_pred = (selected_proba >= best_threshold).astype(int)
    alt_metrics = compute_metrics(y_test, alt_pred, selected_proba)

    # ---------------- Figures ------------------------------------------
    plot_cv_scores(baseline_folds)
    plot_cv_metrics(baseline_cv)
    plot_confusion_matrices(
        {name: (proba >= config.THRESHOLD).astype(int) for name, proba in tuned_probas.items()},
        y_test,
        "Confusion matrices - tuned models (test set, threshold 0.50)",
        "12_confusion_matrices.png",
    )
    plot_roc_pr(tuned_probas, y_test)
    plot_final_confusion(y_test, selected_pred, final_metrics, selected_name)
    importance = plot_importance(selected_pipeline, X_test, y_test, selected_name)
    learning = plot_learning_curve(selected_pipeline, X_train, y_train, cv)
    learning.round(4).to_csv(config.METRICS_DIR / "learning_curve.csv", index=False)

    # ---------------- Save deployable bundle ---------------------------
    medians = X_train.median(numeric_only=True).to_dict()
    input_defaults = {
        "Air temperature [K]": round(float(medians["Air temperature [K]"]), 1),
        "Process temperature [K]": round(float(medians["Process temperature [K]"]), 1),
        "Rotational speed [rpm]": int(round(float(medians["Rotational speed [rpm]"]))),
        "Torque [Nm]": round(float(medians["Torque [Nm]"]), 1),
        "Tool wear [min]": int(round(float(medians["Tool wear [min]"]))),
        "Type": "L",
    }
    input_ranges = {
        column: [float(X_train[column].min()), float(X_train[column].max())]
        for column in config.NUMERIC_FEATURES
    }

    bundle = {
        "pipeline": selected_pipeline,
        "model_name": selected_name,
        "threshold": config.THRESHOLD,
        "features": list(config.FEATURES),
        "engineered_features": list(config.ENGINEERED_FEATURES)
        if config.USE_ENGINEERED_FEATURES else [],
        "input_defaults": input_defaults,
        "input_ranges": input_ranges,
        "best_params": _clean_params(selected_search.best_params_),
        "cv_f1_mean": round(float(tuning_table.set_index("model").loc[selected_name, "best_cv_f1"]), 4),
        "test_metrics": {
            key: (int(value) if key in ("tn", "fp", "fn", "tp") else round(float(value), 4))
            for key, value in final_metrics.items()
        },
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "random_state": config.RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    joblib.dump(bundle, config.MODEL_PATH)

    model_card = {
        "model_name": selected_name,
        "artifact": config.MODEL_PATH.name,
        "task": "Binary classification - industrial machine failure",
        "dataset": "AI4I 2020 Predictive Maintenance (UCI), 10,000 records",
        "features_used": list(config.FEATURES),
        "engineered_features": bundle["engineered_features"],
        "excluded_columns": config.ID_COLUMNS + config.FAILURE_MODE_COLUMNS,
        "hyperparameters": bundle["best_params"],
        "decision_threshold": config.THRESHOLD,
        "cv_f1_mean": bundle["cv_f1_mean"],
        "test_metrics": bundle["test_metrics"],
        "confusion_matrix": {
            "tn": final_metrics["tn"], "fp": final_metrics["fp"],
            "fn": final_metrics["fn"], "tp": final_metrics["tp"],
        },
        "trained_at": bundle["trained_at"],
        "versions": {"scikit-learn": sklearn.__version__,
                     "python": platform.python_version()},
        "intended_use": "Educational predictive-maintenance demo on AI4I-like data.",
        "limitations": [
            "Synthetic AI4I data; not validated on real plant telemetry.",
            "Failure prevalence is 3.4%, so precision/recall matter more than accuracy.",
            "Random failures (RNF flag) are by construction unpredictable.",
        ],
    }
    config.MODEL_CARD_PATH.write_text(json.dumps(model_card, indent=2), encoding="utf-8")

    # ---------------- Report -------------------------------------------
    results = {
        "selected_name": selected_name,
        "selected_params": _clean_params(selected_search.best_params_),
        "baseline_cv": baseline_cv,
        "baseline_test": baseline_test,
        "tuned_test": tuned_test,
        "tuning_table": tuning_table,
        "final_metrics": final_metrics,
        "alt_metrics": alt_metrics,
        "best_threshold": best_threshold,
        "best_threshold_f1": best_threshold_f1,
        "importance": importance,
        "learning": learning,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_fail_train": int(y_train.sum()),
        "n_fail_test": int(y_test.sum()),
        "n_features": int(
            selected_pipeline.named_steps["preprocessor"]
            .named_steps["columns"].get_feature_names_out().shape[0]
        ),
        "elapsed": time.perf_counter() - started,
        "threshold_table": threshold_table,
    }
    write_report(results)

    if verbose:
        print("Final test performance of the selected model")
        print("-" * 74)
        print(f"Accuracy  : {final_metrics['accuracy']:.3f}")
        print(f"Precision : {final_metrics['precision']:.3f}  (failure class)")
        print(f"Recall    : {final_metrics['recall']:.3f}  (failure class)")
        print(f"F1-score  : {final_metrics['f1']:.3f}")
        print(f"ROC-AUC   : {final_metrics['roc_auc']:.3f}")
        print(f"PR-AUC    : {final_metrics['pr_auc']:.3f}")
        print("Confusion matrix [rows actual, cols predicted]:")
        print(f"                 pred No   pred Yes")
        print(f"  actual No    {final_metrics['tn']:9d} {final_metrics['fp']:9d}")
        print(f"  actual Yes   {final_metrics['fn']:9d} {final_metrics['tp']:9d}")
        print(f"\nModel saved to : {config.MODEL_PATH}")
        print(f"Report         : {config.REPORTS_DIR / 'REPORT.md'}")
        print(f"Total runtime  : {results['elapsed']:.1f}s")

    return results


# ----------------------------------------------------------------------
# Report generation
# ----------------------------------------------------------------------
def write_report(results: dict) -> None:
    bcv = results["baseline_cv"]
    btest = results["baseline_test"]
    tt = results["tuning_table"]
    tuned = results["tuned_test"]
    fm = results["final_metrics"]
    alt = results["alt_metrics"]
    name = results["selected_name"]

    cv_table = bcv.reset_index()[
        ["model", "f1_mean", "f1_std", "recall_mean", "precision_mean", "accuracy_mean"]
    ].copy()
    cv_table.columns = ["Model", "CV F1 (mean)", "Std", "CV recall", "CV precision", "CV accuracy"]

    tuning_view = tt.copy()
    tuning_view["best_params"] = tuning_view["best_params"].apply(
        lambda value: "`" + value.replace('"', "'").replace("null", "None") + "`"
    )
    tuning_view = tuning_view[
        ["model", "best_cv_f1", "best_cv_f1_std", "best_cv_recall", "best_params",
         "candidates_tested", "search_seconds"]
    ].rename(columns={
        "model": "Model", "best_cv_f1": "Best CV F1", "best_cv_f1_std": "Std",
        "best_cv_recall": "CV recall", "best_params": "Best hyperparameters",
        "candidates_tested": "Combinations", "search_seconds": "Seconds",
    })

    test_view = tuned[["model"] + METRIC_COLUMNS].rename(
        columns={"model": "Model", **PRETTY}
    )

    baseline_view = btest[["model"] + METRIC_COLUMNS].rename(
        columns={"model": "Model", **PRETTY}
    )
    baseline_md = markdown_table(baseline_view.round(3))
    cv_md = markdown_table(cv_table.round(3))
    tuned_md = markdown_table(test_view.round(3))

    cm_rows = [
        ["Actual No", fm["tn"], fm["fp"]],
        ["Actual Yes", fm["fn"], fm["tp"]],
    ]
    cm_table = pd.DataFrame(cm_rows, columns=["", "Predicted No", "Predicted Yes"])
    cm_md = markdown_table(cm_table)

    top_features = results["importance"]
    feature_lines = "\n".join(
        f"| {row.feature} | {row.permutation_importance:.3f} |"
        for row in top_features.itertuples()
    )

    params_text = ", ".join(
        f"{key}={value}" for key, value in results["selected_params"].items()
    )

    report = f"""# Industrial Machine Failure Prediction

**Dataset:** AI4I 2020 Predictive Maintenance (UCI) - 10,000 records, 14 columns
**Task:** binary classification of `Machine failure` (Yes / No)
**Selected model:** **{name}** (`{params_text}`)
**Trained:** {datetime.now().strftime('%d %B %Y, %H:%M')} | scikit-learn {sklearn.__version__}

---

## 1. Dataset understanding

The AI4I dataset simulates 10,000 industrial machines described by a quality grade
(`Type`: L/M/H), three thermal/kinematic sensor families
(air temperature, process temperature, rotational speed, torque) and accumulated
`Tool wear`. 339 machines failed (3.39%), so this is a strongly imbalanced
binary classification problem.

| Column | Role |
|---|---|
| `UDI`, `Product ID` | row identifiers - removed |
| `Type` | categorical input (one-hot encoded) |
| `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]` | numeric inputs |
| `Machine failure` | target |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | failure-mode flags describing *why* the machine failed - removed to prevent leakage |

## 2. Exploratory data analysis

No missing values, no duplicate rows, all 10,000 records usable.
The target is imbalanced (96.61% healthy / 3.39% failure), which makes
**accuracy a misleading metric**: a model that always says "No" already scores 96.6%.

Key findings (see `reports/figures/01`-`09`):

- Failing machines show **higher torque, higher tool wear and lower rotational
  speed** than healthy machines; air temperature is slightly higher.
- Failures cluster at the boundary of the torque/speed envelope, i.e. at high
  mechanical power (`06_torque_rpm_power.png`).
- `Type` matters mildly: failure rates differ across L/M/H, and low-quality
  machines dominate the fleet.
- The five failure-mode flags are near-perfect predictors of the target because
  they *compose* it (a simple OR of the flags reproduces `Machine failure` for
  99.73% of rows). Including them would produce meaningless ~100% accuracy,
  so they are excluded from `X` (`04_correlation_heatmap.png`, right panel).

Descriptive tables: `reports/metrics/eda_*.csv`.

![Class distribution](figures/01_class_distribution.png)

![Torque vs speed](figures/06_torque_rpm_power.png)

## 3. Data preparation

- Removed `UDI`, `Product ID` (identifiers) and `TWF`, `HDF`, `PWF`, `OSF`, `RNF`
  (target-derived leakage).
- Missing values: none; the pipeline nevertheless contains median / most-frequent
  imputers so it is safe on new data.
- Encoded `Type` with one-hot encoding (`L` is the reference category).
- Added two physically motivated features (no leakage, inputs only):
  - `Temperature difference [K] = process - air`
  - `Mechanical power [W] = torque * 2*pi*rpm / 60`
- Scaled numeric features with `StandardScaler` for Logistic Regression
  (harmless for trees).
- All preprocessing lives in a scikit-learn `Pipeline`, so the exact same
  transformations are applied at prediction time.

Final feature space: **{results['n_features']} columns** after encoding.

## 4. Train / test split

Stratified {int(100 * (1 - config.TEST_SIZE))}/{int(100 * config.TEST_SIZE)} split,
`random_state={config.RANDOM_STATE}`:

| Set | Records | Failures | Prevalence |
|---|---|---|---|
| Training | {results['n_train']:,} | {results['n_fail_train']} | {100 * results['n_fail_train'] / results['n_train']:.2f}% |
| Test | {results['n_test']:,} | {results['n_fail_test']} | {100 * results['n_fail_test'] / results['n_test']:.2f}% |

The test set is not used for fitting or model selection; it is only used for the
comparison tables and the final evaluation reported here.

## 5-6. Baseline models and comparison

Logistic Regression, Decision Tree and Random Forest were trained with
class-imbalance-aware settings (balanced class weights). Full metric table on the
test set (threshold 0.50):

{baseline_md}

Accuracy looks excellent for the tree models, but on a 96.6% healthy dataset a
constant "No failure" prediction already scores 96.6%, so precision, recall and
F1 for the failure class are the metrics that matter.

## 7. 5-fold cross-validation

Stratified 5-fold CV was run on the training data only, so the test set stays
untouched (per-fold scores in `reports/metrics/cv_baseline_folds.csv`):

{cv_md}

![5-fold CV F1 scores](figures/10_cv_f1_scores.png)

![CV metric comparison](figures/11_cv_metrics_comparison.png)

Random Forest has the highest mean CV F1 ({bcv.loc['Random Forest', 'f1_mean']:.3f}).
The Decision Tree is close but less consistent across folds, the expected
signature of a single high-variance tree. Logistic Regression is the most stable
model but also the weakest: class weighting gives it high recall at very low
precision, so its F1 stays around {bcv.loc['Logistic Regression', 'f1_mean']:.2f}.

## 8. Hyperparameter tuning

`GridSearchCV` with the same stratified 5-fold split, optimising **F1 of the
failure class** (a sensible choice under 3.4% prevalence).

{markdown_table(tuning_view)}

The complete grids are stored in `reports/metrics/grid_*.csv`; the tuning table
with all metrics is `reports/metrics/tuning_results.csv`.

Note that tuning maximises F1 at a fixed 0.50 threshold, not recall. For
Logistic Regression the grid preferred an unweighted model that is more precise
but misses most failures, which is why its recall falls compared with the
balanced baseline. This shows why the metric and the operating point have to be
chosen together.

## 9. Model selection and final evaluation

Model selection is based on **cross-validated F1** (tie-break: CV recall), not on
the test set. Selected: **{name}**.

All three tuned models on the untouched test set:

{tuned_md}

![Tuned model confusion matrices](figures/12_confusion_matrices.png)

![ROC and PR curves](figures/13_roc_pr_curves.png)

### Final test metrics

| Metric | Value |
|---|---|
| Accuracy | {fm['accuracy']:.4f} |
| Precision (failure) | {fm['precision']:.4f} |
| Recall (failure) | {fm['recall']:.4f} |
| F1-score (failure) | {fm['f1']:.4f} |
| ROC-AUC | {fm['roc_auc']:.4f} |
| PR-AUC | {fm['pr_auc']:.4f} |

### Confusion matrix (test set, threshold 0.50)

{cm_md}

![Confusion matrix of the selected model](figures/17_final_confusion_matrix.png)

- Caught **{fm['tp']} of {fm['tp'] + fm['fn']}** failures
  (recall {fm['recall']:.1%}).
- **{fm['fp']}** false alarms out of {fm['tn'] + fm['fp']} healthy machines
  (false-positive rate {fm['fp'] / (fm['tn'] + fm['fp']):.1%}).
- Missed failures ({fm['fn']}) are the expensive error for maintenance planning,
  which is why F1 (which balances recall against false alarms) was optimised.

### Operating point

Out-of-fold predictions show that the F1-optimal threshold is
**{results['best_threshold']:.2f}** (F1 {results['best_threshold_f1']:.3f} on training).
For reference, applying it to the test set gives
precision {alt['precision']:.3f} / recall {alt['recall']:.3f} / F1 {alt['f1']:.3f}
versus {fm['precision']:.3f} / {fm['recall']:.3f} / {fm['f1']:.3f} at 0.50.
The deployed system keeps 0.50 so that predicted probabilities stay directly
interpretable; `15_threshold_analysis.png` documents the trade-off.

### What drives the prediction

| Feature | Permutation importance (drop in test F1) |
|---|---|
{feature_lines}

![Feature importance](figures/14_feature_importance.png)

The model's internal (impurity) importance concentrates on the derived
**mechanical power**, then rotational speed and torque. Permutation importance
spreads the credit differently because power is a deterministic function of
torque and speed, so shuffling any one of them damages a signal the tree can
partly rebuild from the others. Both views agree with the EDA: power, torque,
speed and tool wear are the physical drivers of failure in this dataset, while
`Type` contributes little.

![Learning curve](figures/16_learning_curve.png)

The learning curve flattens as training size grows, so extra data of the same
kind would bring only modest F1 gains.

## 10-11. Prediction interface and saved artifacts

- `models/machine_failure_model.joblib` - complete pipeline (engineering +
  encoding + scaling + classifier) and metadata.
- `models/model_card.json` - machine-readable model card.
- `python main.py predict` - interactive command-line interface.
- `python main.py predict --air 298.1 --process 308.6 --rpm 1550 --torque 42.8 --wear 0 --type L --explain`
- `streamlit run app.py` - web form.

The prediction path loads the artifact, validates inputs against the training
ranges, and returns `Machine Failure: Yes/No` plus the estimated probability.

## Limitations

- AI4I is synthetic: failure modes are generated from threshold rules plus
  random failures, so real plant data would be noisier.
- 339 positive cases limit how precise the performance estimates are; CV std
  and the confidence of the test metrics should be read together.
- Random failures (RNF) are unpredictable by construction and cap achievable recall.
- The model predicts *risk from current operating conditions*, not remaining
  useful life.
"""
    (config.REPORTS_DIR / "REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    run_training()
