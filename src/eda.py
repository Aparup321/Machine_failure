"""Step 2 - exploratory data analysis.

Produces the printed summary, the statistics tables and every EDA figure used
in the final report. Everything is written into reports/.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from .data_loader import describe_dataset, load_raw_data
from .plotting import PALETTE, MODEL_COLORS, save_figure


def run_eda(verbose: bool = True) -> dict:
    raw = load_raw_data()
    df = raw.drop(columns=config.ID_COLUMNS)

    summary = describe_dataset(raw)
    healthy = df[df[config.TARGET] == 0]
    failed = df[df[config.TARGET] == 1]

    # ------------------------------------------------------------------
    # Numeric summary tables
    # ------------------------------------------------------------------
    numeric_stats = df[config.NUMERIC_FEATURES].describe().T
    numeric_stats["median"] = df[config.NUMERIC_FEATURES].median()
    numeric_stats["skew"] = df[config.NUMERIC_FEATURES].skew()

    by_failure = df.groupby(config.TARGET)[config.NUMERIC_FEATURES].mean().T
    by_failure.columns = ["Healthy mean", "Failure mean"]

    type_table = (
        df.groupby("Type")
        .agg(machines=(config.TARGET, "size"), failures=(config.TARGET, "sum"))
        .assign(failure_rate_pct=lambda t: 100 * t["failures"] / t["machines"])
        .reindex(config.TYPE_CATEGORIES)
    )

    derived = df.copy()
    derived["Mechanical power [W]"] = (
        derived["Torque [Nm]"] * 2 * np.pi * derived["Rotational speed [rpm]"] / 60
    )
    derived["Temperature difference [K]"] = (
        derived["Process temperature [K]"] - derived["Air temperature [K]"]
    )
    derived_stats = derived.groupby(config.TARGET)[
        ["Mechanical power [W]", "Temperature difference [K]", "Torque [Nm]", "Tool wear [min]"]
    ].median().T
    derived_stats.columns = ["Healthy median", "Failure median"]

    numeric_stats.to_csv(config.METRICS_DIR / "eda_numeric_summary.csv")
    by_failure.to_csv(config.METRICS_DIR / "eda_feature_means_by_failure.csv")
    type_table.to_csv(config.METRICS_DIR / "eda_failure_rate_by_type.csv")
    derived_stats.to_csv(config.METRICS_DIR / "eda_engineered_medians.csv")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    _fig_class_balance(df)
    _fig_numeric_distributions(df, failed)
    _fig_boxplots(df)
    _fig_correlation(df, raw)
    _fig_failure_rate_by_type(df)
    _fig_torque_rpm(df)
    _fig_wear_torque(df)
    _fig_failure_modes(raw)
    _fig_engineered(df, derived)

    if verbose:
        _print_eda(summary, numeric_stats, by_failure, type_table, derived_stats, raw)

    return {
        "summary": summary,
        "numeric_stats": numeric_stats.round(3).to_dict(),
        "means_by_failure": by_failure.round(3).to_dict(),
        "failure_rate_by_type": type_table.round(3).to_dict(),
        "engineered_medians": derived_stats.round(2).to_dict(),
    }


# ----------------------------------------------------------------------
# Individual figures
# ----------------------------------------------------------------------
def _fig_class_balance(df: pd.DataFrame) -> None:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    bars = axes[0].bar(
        ["No failure", "Failure"], counts.values,
        color=[PALETTE[0], PALETTE[1]], width=0.55,
    )
    axes[0].set_title("Class distribution")
    axes[0].set_ylabel("Number of machines")
    for bar, value in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 120,
            f"{value:,}\n({100 * value / len(df):.2f}%)",
            ha="center", va="bottom", fontsize=9,
        )
    axes[0].set_ylim(0, counts.max() * 1.18)

    axes[1].pie(
        counts.values,
        labels=["No failure", "Failure"],
        colors=[PALETTE[0], PALETTE[1]],
        autopct="%1.1f%%",
        startangle=120,
        explode=(0, 0.12),
        textprops={"fontsize": 9},
    )
    axes[1].set_title("Failure share (imbalanced target)")
    fig.suptitle("Target variable: Machine failure", fontsize=13, fontweight="bold")
    save_figure(fig, "01_class_distribution.png")


def _fig_numeric_distributions(df: pd.DataFrame, failed: pd.DataFrame) -> None:
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.1))
    for ax, column in zip(axes, config.NUMERIC_FEATURES):
        bins = np.histogram_bin_edges(df[column], bins=32)
        for label, colour in ((0, PALETTE[0]), (1, PALETTE[1])):
            subset = df.loc[df[config.TARGET] == label, column]
            ax.hist(subset, bins=bins, density=True, histtype="stepfilled",
                    alpha=0.28, color=colour)
            ax.hist(subset, bins=bins, density=True, histtype="step",
                    linewidth=1.1, color=colour)
            ax.axvline(subset.mean(), color=colour, linestyle="--", linewidth=1.1)
        ax.set_title(column, fontsize=9.5)
        ax.set_ylabel("Density (per class)")
        ax.tick_params(labelsize=8)

    handles = [
        Patch(facecolor=PALETTE[0], alpha=0.5, label="No failure"),
        Patch(facecolor=PALETTE[1], alpha=0.5, label="Failure"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=2, fontsize=9)
    fig.suptitle("Sensor distributions by machine status "
                 "(dashed line = class mean, densities normalised per class)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "02_numeric_distributions.png")


def _fig_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, column in zip(axes, config.NUMERIC_FEATURES):
        sns.boxplot(
            data=df, x=config.TARGET, y=column, ax=ax,
            hue=config.TARGET, palette=PALETTE, legend=False, width=0.55,
            fliersize=2,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No", "Yes"])
        ax.set_xlabel("Machine failure")
        ax.set_title(column.replace(" [K]", "\n[K]").replace(" [rpm]", "\n[rpm]")
                     .replace(" [Nm]", "\n[Nm]").replace(" [min]", "\n[min]"), fontsize=9)
    fig.suptitle("Measurement spread for healthy vs failing machines", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "03_boxplots_by_failure.png")


def _fig_correlation(df: pd.DataFrame, raw: pd.DataFrame) -> None:
    columns = config.NUMERIC_FEATURES + [config.TARGET]
    corr = df[columns].corr(numeric_only=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[0],
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
        annot_kws={"fontsize": 7.5},
    )
    axes[0].set_title("Correlation with the failure target")
    axes[0].tick_params(labelsize=7.5)

    # Leakage check: each failure-mode flag vs. the recorded failure
    modes = raw[config.FAILURE_MODE_COLUMNS + [config.TARGET]].corr()
    sns.heatmap(
        modes, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[1],
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
        annot_kws={"fontsize": 8},
    )
    axes[1].set_title("Failure-mode flags vs target (leakage!)")
    axes[1].tick_params(labelsize=8)
    fig.tight_layout()
    save_figure(fig, "04_correlation_heatmap.png")


def _fig_failure_rate_by_type(df: pd.DataFrame) -> None:
    table = (
        df.groupby("Type")[config.TARGET]
        .agg(["size", "sum", "mean"])
        .reindex(config.TYPE_CATEGORIES)
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    x = np.arange(len(table))
    axes[0].bar(x, table["size"], color=MODEL_COLORS[:3], width=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["L (low)", "M (medium)", "H (high)"])
    axes[0].set_ylabel("Machines")
    axes[0].set_title("Machine quality mix")
    for xi, value in zip(x, table["size"]):
        axes[0].text(xi, value + 80, f"{value:,}", ha="center", fontsize=9)

    axes[1].bar(x, 100 * table["mean"], color=MODEL_COLORS[:3], width=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["L (low)", "M (medium)", "H (high)"])
    axes[1].set_ylabel("Failure rate (%)")
    axes[1].set_title("Failure rate by machine quality")
    for xi, value in zip(x, 100 * table["mean"]):
        axes[1].text(xi, value + 0.05, f"{value:.2f}%", ha="center", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "05_type_vs_failure.png")


def _fig_torque_rpm(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    healthy = df[df[config.TARGET] == 0]
    failed = df[df[config.TARGET] == 1]

    ax.scatter(healthy["Rotational speed [rpm]"], healthy["Torque [Nm]"],
               s=8, alpha=0.35, color=PALETTE[0], label="No failure")
    ax.scatter(failed["Rotational speed [rpm]"], failed["Torque [Nm]"],
               s=26, alpha=0.9, color=PALETTE[1], edgecolor="black",
               linewidth=0.4, label="Failure")
    ax.set_xlabel("Rotational speed [rpm]")
    ax.set_ylabel("Torque [Nm]")
    ax.set_title("Failures live at the edge of the torque / speed envelope")
    ax.legend(loc="upper right")

    # iso-power curves: P = torque * 2*pi*rpm/60
    rpm_grid = np.linspace(1100, 2900, 200)
    for power in (4000, 6000, 8000, 10000):
        ax.plot(rpm_grid, power * 60 / (2 * np.pi * rpm_grid),
                linestyle="--", linewidth=0.9, color="gray", alpha=0.7)
        ax.text(rpm_grid[-1] - 20, power * 60 / (2 * np.pi * rpm_grid[-1]),
                f"{power // 1000} kW", fontsize=7, color="gray", va="bottom", ha="right")
    fig.tight_layout()
    save_figure(fig, "06_torque_rpm_power.png")


def _fig_wear_torque(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    failed = df[df[config.TARGET] == 1]
    healthy = df.sample(min(2500, len(df)), random_state=config.RANDOM_STATE)

    ax.scatter(healthy["Tool wear [min]"], healthy["Torque [Nm]"],
               s=8, alpha=0.3, color=PALETTE[0], label="No failure (sample)")
    ax.scatter(failed["Tool wear [min]"], failed["Torque [Nm]"],
               s=26, alpha=0.9, color=PALETTE[1], edgecolor="black",
               linewidth=0.4, label="Failure")
    ax.set_xlabel("Tool wear [min]")
    ax.set_ylabel("Torque [Nm]")
    ax.set_title("Tool wear and torque together drive failures")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, "07_wear_torque.png")


def _fig_failure_modes(raw: pd.DataFrame) -> None:
    counts = raw[config.FAILURE_MODE_COLUMNS].sum()
    overlaps = raw[config.FAILURE_MODE_COLUMNS].sum(axis=1).value_counts().sort_index()
    agreement = (
        raw[config.FAILURE_MODE_COLUMNS].max(axis=1) == raw[config.TARGET]
    ).mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    bars = axes[0].bar(counts.index, counts.values, color=MODEL_COLORS[:5], width=0.6)
    axes[0].set_title("Failure-mode indicator counts")
    axes[0].set_ylabel("Occurrences")
    for bar, value in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 2, str(value),
                     ha="center", fontsize=9)

    axes[1].bar(overlaps.index.astype(str), overlaps.values, color="#457B9D", width=0.55)
    axes[1].set_xlabel("Number of simultaneously active failure modes")
    axes[1].set_ylabel("Machines")
    axes[1].set_title("Recorded failures can have several modes")
    for xi, value in zip(range(len(overlaps)), overlaps.values):
        axes[1].text(xi, value + 18, str(value), ha="center", fontsize=9)
    fig.suptitle(
        f"These five columns encode the failure (OR of flags = target for "
        f"{100 * agreement:.2f}% of rows) - excluded from X",
        fontsize=12.5, fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "08_failure_modes.png")


def _fig_engineered(df: pd.DataFrame, derived: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, column, unit in [
        (axes[0], "Mechanical power [W]", "W"),
        (axes[1], "Temperature difference [K]", "K"),
    ]:
        sns.violinplot(
            data=derived, x=config.TARGET, y=column, hue=config.TARGET,
            palette=PALETTE, ax=ax, legend=False, inner="quartile",
            cut=0, linewidth=0.8,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No failure", "Failure"])
        ax.set_xlabel("")
        ax.set_ylabel(unit)
        ax.set_title(column)
    fig.suptitle("Engineered features separate the classes better than raw sensors",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "09_engineered_features.png")


# ----------------------------------------------------------------------
def _print_eda(summary, numeric_stats, by_failure, type_table, derived_stats, raw) -> None:
    print("=" * 74)
    print("EXPLORATORY DATA ANALYSIS".center(74))
    print("=" * 74)
    print(f"Rows: {summary['n_rows']:,} | Missing: {summary['missing_total']} | "
          f"Duplicates: {summary['duplicate_rows']}")
    print(f"Failures: {summary['n_failures']} ({summary['failure_rate_pct']}%) | "
          f"Healthy: {summary['n_healthy']}")
    print("\nFailure-mode indicator counts (excluded from the model):")
    for mode, count in summary["failure_modes"].items():
        print(f"  {mode}: {count}")

    print("\nNumeric summary:")
    print(numeric_stats.round(2).to_string())
    print("\nMean sensor values by machine status:")
    print(by_failure.round(2).to_string())
    print("\nFailure rate by machine quality:")
    print(type_table.round(2).to_string())
    print("\nMedian of engineered features by machine status:")
    print(derived_stats.round(1).to_string())
    print("\nFigures written to:", config.FIGURES_DIR)
    print()
