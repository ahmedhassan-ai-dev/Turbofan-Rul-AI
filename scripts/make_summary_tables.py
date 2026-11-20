"""Generate the markdown tables used in the README and the report.

Every table is derived from files on disk (registry, run summaries,
official test results). Nothing is typed by hand, so the documentation
cannot drift away from what was actually run.

Usage
-----
    python scripts/make_summary_tables.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.registry import read_registry  # noqa: E402
from src.reporting.tables import to_markdown  # noqa: E402


OUT_DIR = ROOT / "experiments" / "results" / "summary_tables"

TEST_DIR = ROOT / "experiments" / "results" / "official_test"
COST_DIR = ROOT / "experiments" / "results" / "cost"


def validation_table(registry: pd.DataFrame) -> pd.DataFrame:
    main = registry[registry["tag"] == "main"].copy()

    if main.empty:
        return pd.DataFrame()

    main = main.sort_values(["model", "seed"])

    grouped = (
        main.groupby("model")["val_rmse"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={
            "mean": "val_rmse_mean",
            "std": "val_rmse_std",
            "min": "val_rmse_min",
            "max": "val_rmse_max",
            "count": "seeds",
        })
    )

    return grouped.sort_values("val_rmse_mean")


def full_registry_table(registry: pd.DataFrame) -> pd.DataFrame:
    if registry.empty:
        return pd.DataFrame()

    columns = [
        "experiment_id", "tag", "model", "seed", "window_size",
        "rul_cap", "learning_rate", "batch_size", "epochs_ran",
        "best_epoch", "val_rmse", "val_mae", "val_bias",
        "train_seconds", "status",
    ]

    return registry[columns].sort_values(
        ["tag", "model", "seed"]
    )


def test_table() -> pd.DataFrame:
    path = TEST_DIR / "test_metrics.csv"

    if not path.is_file():
        return pd.DataFrame()

    return pd.read_csv(path).sort_values("rmse")


def cost_table() -> pd.DataFrame:
    path = COST_DIR / "cost_table.csv"

    if not path.is_file():
        return pd.DataFrame()

    return pd.read_csv(path)


def ablation_table(registry: pd.DataFrame) -> pd.DataFrame:
    hybrid = registry[
        (registry["tag"] == "main")
        & (registry["model"] == "cnn_bilstm_attention")
    ]
    mean = registry[
        (registry["tag"] == "ablation")
        & (registry["model"] == "cnn_bilstm_mean")
    ]

    if hybrid.empty or mean.empty:
        return pd.DataFrame()

    rows = []

    for label, group in [
        ("CNN-BiLSTM + attention", hybrid),
        ("CNN-BiLSTM + mean pooling", mean),
    ]:
        rows.append({
            "variant": label,
            "seeds": len(group),
            "val_rmse_mean": group["val_rmse"].mean(),
            "val_rmse_std": group["val_rmse"].std(),
            "val_mae_mean": group["val_mae"].mean(),
            "val_nasa_mean": group["val_nasa_score_mean"].mean(),
        })

    return pd.DataFrame(rows)


def horizon_table() -> pd.DataFrame:
    path = (
        ROOT / "experiments" / "results" / "error_analysis"
        / "metrics_by_horizon.csv"
    )

    if not path.is_file():
        return pd.DataFrame()

    pivot = pd.read_csv(path).pivot(
        index="horizon", columns="model", values="rmse"
    ).reset_index()

    return pivot


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    registry = read_registry()

    tables = {
        "validation_summary.md": (
            "Validation summary per model (main runs, 3 seeds)",
            validation_table(registry),
        ),
        "experiment_registry.md": (
            "All registered experiments",
            full_registry_table(registry),
        ),
        "official_test.md": (
            "Official FD001 test results",
            test_table(),
        ),
        "attention_ablation.md": (
            "Attention ablation (identical architecture, "
            "mean pooling instead of attention)",
            ablation_table(registry),
        ),
        "cost.md": (
            "Computational cost",
            cost_table(),
        ),
        "error_by_horizon.md": (
            "RMSE by RUL horizon (official test)",
            horizon_table(),
        ),
    }

    for filename, (title, table) in tables.items():
        if table.empty:
            content = f"# {title}\n\n_Pending: no data available yet._\n"
        else:
            content = f"# {title}\n\n{to_markdown(table)}\n"

        (OUT_DIR / filename).write_text(content, encoding="utf-8")

        print(f"- {filename}: {'empty' if table.empty else len(table)} rows")

    (OUT_DIR / "all_tables.md").write_text(
        "\n\n".join(
            (OUT_DIR / filename).read_text(encoding="utf-8")
            for filename in tables
        ),
        encoding="utf-8",
    )

    print("\nWritten to:", OUT_DIR)


if __name__ == "__main__":
    main()
