"""Error analysis on the official FD001 test predictions.

Every figure and number here is computed from the prediction files
written by ``src.evaluation.test_evaluation``. Nothing is hard-coded.

RUL horizons (chosen from the FD001 test label distribution and stated
in the report):

* Short  : true RUL <= 50 cycles
* Medium : 50 < true RUL <= 100 cycles
* Long   : true RUL > 100 cycles
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import regression_metrics
from src.reporting.tables import to_markdown


ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = ROOT / "experiments" / "results" / "official_test"
FIGURES_DIR = ROOT / "experiments" / "figures" / "error_analysis"
TABLES_DIR = ROOT / "experiments" / "results" / "error_analysis"

HORIZON_BINS = [
    ("Short (<=50)", 0, 50),
    ("Medium (51-100)", 50, 100),
    ("Long (>100)", 100, np.inf),
]


def load_predictions() -> dict[str, pd.DataFrame]:
    if not PREDICTIONS_DIR.is_dir():
        raise FileNotFoundError(
            "No official test predictions. Run "
            "`python -m src.evaluation.test_evaluation --auto` first."
        )

    tables = {}

    for path in sorted(PREDICTIONS_DIR.glob("*_predictions.csv")):
        label = path.name.replace("_predictions.csv", "")
        table = pd.read_csv(path)

        required = {
            "engine_id", "true_rul", "predicted_rul", "error"
        }

        if not required.issubset(table.columns):
            raise ValueError(
                f"{path.name} is missing columns: "
                f"{sorted(required - set(table.columns))}"
            )

        tables[label] = table

    if not tables:
        raise FileNotFoundError(
            f"No *_predictions.csv files inside {PREDICTIONS_DIR}"
        )

    return tables


def horizon_of(rul: float) -> str:
    for name, low, high in HORIZON_BINS:
        if low < rul <= high or (low == 0 and rul <= high):
            return name

    return HORIZON_BINS[-1][0]


def plot_actual_vs_predicted(tables: dict[str, pd.DataFrame]) -> None:
    n_models = len(tables)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(5.2 * n_models, 4.8),
        squeeze=False,
    )

    for ax, (label, table) in zip(axes[0], tables.items()):
        ax.scatter(
            table["true_rul"],
            table["predicted_rul"],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.4,
        )

        limit = max(
            table["true_rul"].max(),
            table["predicted_rul"].max(),
        ) * 1.05

        ax.plot([0, limit], [0, limit], "--", color="grey")
        ax.set_title(label)
        ax.set_xlabel("True RUL")
        ax.set_ylabel("Predicted RUL")

    fig.suptitle("Actual vs predicted RUL — official FD001 test")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "actual_vs_predicted_all_models.png",
        dpi=180,
    )
    plt.close(fig)


def plot_residuals(tables: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, table in tables.items():
        ax.hist(
            table["error"],
            bins=25,
            alpha=0.55,
            label=label,
        )

    ax.axvline(0, linestyle="--", color="black")
    ax.set_xlabel("Prediction error (predicted - true, cycles)")
    ax.set_ylabel("Number of engines")
    ax.set_title("Residual distribution — official FD001 test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "residual_distribution.png",
        dpi=180,
    )
    plt.close(fig)


def plot_error_vs_actual(tables: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))

    for label, table in tables.items():
        ax.scatter(
            table["true_rul"],
            table["error"],
            alpha=0.65,
            label=label,
        )

    ax.axhline(0, linestyle="--", color="black")
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Error (predicted - true)")
    ax.set_title("Error vs true RUL — official FD001 test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "error_vs_actual_rul.png",
        dpi=180,
    )
    plt.close(fig)


def plot_horizon_errors(
    horizon_table: pd.DataFrame,
) -> None:
    if horizon_table.empty:
        return

    pivot = horizon_table.pivot(
        index="horizon",
        columns="model",
        values="rmse",
    )

    order = [name for name, _, _ in HORIZON_BINS]
    pivot = pivot.reindex([h for h in order if h in pivot.index])

    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_xlabel("True RUL horizon")
    ax.set_ylabel("RMSE (cycles)")
    ax.set_title("RMSE by RUL horizon — official FD001 test")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "error_by_horizon.png",
        dpi=180,
    )
    plt.close()


def plot_model_comparison(
    metrics_table: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    metrics_table.plot(
        kind="bar", x="model", y="rmse", ax=axes[0],
        legend=False, color="#4C72B0",
    )
    axes[0].set_title("RMSE (lower is better)")
    axes[0].set_ylabel("cycles")
    axes[0].set_xlabel("")

    metrics_table.plot(
        kind="bar", x="model", y="nasa_score_mean",
        ax=axes[1], legend=False, color="#C44E52",
    )
    axes[1].set_title("Mean NASA score (lower is better)")
    axes[1].set_ylabel("score")
    axes[1].set_xlabel("")

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Model comparison — official FD001 test")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "model_comparison.png",
        dpi=180,
    )
    plt.close(fig)


def plot_engine_errors(tables: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for label, table in tables.items():
        ordered = table.sort_values("error")

        ax.plot(
            np.arange(len(ordered)),
            ordered["error"],
            marker="o",
            markersize=3,
            linewidth=1,
            label=label,
        )

    ax.axhline(0, linestyle="--", color="black")
    ax.set_xlabel("Test engine (sorted by error)")
    ax.set_ylabel("Error (cycles)")
    ax.set_title("Engine-level errors — official FD001 test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "engine_level_errors.png",
        dpi=180,
    )
    plt.close(fig)


def build_markdown(
    metrics_table: pd.DataFrame,
    horizon_table: pd.DataFrame,
    worst: pd.DataFrame,
) -> str:
    lines = [
        "# Error Analysis — Official FD001 Test",
        "",
        "All values below are computed from "
        "`experiments/results/official_test/*_predictions.csv`.",
        "",
        "Horizon definition: Short <= 50 cycles, Medium 51-100, "
        "Long > 100 cycles.",
        "",
        "## Overall metrics",
        "",
        to_markdown(metrics_table),
        "",
        "## RMSE by RUL horizon",
        "",
        (
            to_markdown(horizon_table)
            if not horizon_table.empty
            else "No horizon table."
        ),
        "",
        "## Worst 10 predictions (per model, by absolute error)",
        "",
        to_markdown(worst),
        "",
        "## How to read this",
        "",
        "* Positive error means the model predicted more remaining life "
        "than the engine had (late prediction), which the NASA score "
        "penalises more heavily.",
        "* Horizon tables use the *true* RUL, not the prediction, so a "
        "model cannot inflate its short-horizon score by predicting a "
        "small value everywhere.",
        "",
    ]

    return "\n".join(lines)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    tables = load_predictions()

    rows = []
    horizon_rows = []
    worst_rows = []

    for label, table in tables.items():
        metrics = regression_metrics(
            table["true_rul"], table["predicted_rul"]
        )

        rows.append({
            "model": label,
            **{k: metrics[k] for k in (
                "rmse", "mae", "bias", "r2",
                "nasa_score", "nasa_score_mean",
            )},
        })

        table = table.copy()
        table["horizon"] = table["true_rul"].apply(horizon_of)

        for horizon, group in table.groupby("horizon"):
            horizon_metrics = regression_metrics(
                group["true_rul"], group["predicted_rul"]
            )

            horizon_rows.append({
                "model": label,
                "horizon": horizon,
                "n_engines": len(group),
                "rmse": horizon_metrics["rmse"],
                "mae": horizon_metrics["mae"],
                "bias": horizon_metrics["bias"],
            })

        top = table.reindex(
            table["error"].abs().sort_values(ascending=False).index
        ).head(10)

        for row in top.itertuples(index=False):
            worst_rows.append({
                "model": label,
                "engine_id": int(row.engine_id),
                "true_rul": float(row.true_rul),
                "predicted_rul": float(row.predicted_rul),
                "error": float(row.error),
                "horizon": row.horizon,
            })

    metrics_table = pd.DataFrame(rows).sort_values("rmse")
    horizon_table = pd.DataFrame(horizon_rows)
    worst = pd.DataFrame(worst_rows)

    metrics_table.to_csv(
        TABLES_DIR / "metrics_by_model.csv", index=False
    )
    horizon_table.to_csv(
        TABLES_DIR / "metrics_by_horizon.csv", index=False
    )
    worst.to_csv(
        TABLES_DIR / "worst_10_predictions.csv", index=False
    )

    plot_actual_vs_predicted(tables)
    plot_residuals(tables)
    plot_error_vs_actual(tables)
    plot_horizon_errors(horizon_table)
    plot_model_comparison(metrics_table)
    plot_engine_errors(tables)

    (TABLES_DIR / "error_analysis.md").write_text(
        build_markdown(metrics_table, horizon_table, worst),
        encoding="utf-8",
    )

    (TABLES_DIR / "summary.json").write_text(
        json.dumps({
            "models": list(tables),
            "metrics": rows,
            "horizons": horizon_rows,
        }, indent=2),
        encoding="utf-8",
    )

    print("=== ERROR ANALYSIS ===")
    print(metrics_table.to_string(index=False))
    print("\nFigures:", FIGURES_DIR)
    print("Tables:", TABLES_DIR)


if __name__ == "__main__":
    main()
