
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "experiments" / "figures" / "baseline"

FIGURES.mkdir(parents=True, exist_ok=True)


def metrics(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "bias": np.mean(y_pred - y_true),
    }


def main():
    df = pd.read_csv(
        RESULTS / "random_forest_predictions.csv"
    )

    train = pd.read_csv(
        ROOT / "data/processed/train_FD001.csv"
    )

    y_true = df["true_rul"].to_numpy()
    y_pred = df["predicted_rul"].to_numpy()

    # Simple reference learned from training targets only.
    train_mean = train["rul"].mean()
    constant_predictions = np.full_like(
        y_true, train_mean, dtype=float
    )

    print("\n=== BASELINE COMPARISON ===")
    print("Random Forest:", metrics(y_true, y_pred))
    print("Train-mean predictor:", metrics(
        y_true, constant_predictions
    ))

    print("\n=== PREDICTION DISTRIBUTION ===")
    print(df[["true_rul", "predicted_rul"]].describe())

    print("\n=== ERROR BY HORIZON ===")
    for horizon, group in df.groupby("true_rul"):
        result = metrics(
            group["true_rul"],
            group["predicted_rul"]
        )
        print(f"RUL {horizon:.0f}: {result}")

    df["absolute_error"] = (
        df["predicted_rul"] - df["true_rul"]
    ).abs()

    print("\n=== TOP 10 LARGEST ERRORS ===")
    print(
        df.nlargest(10, "absolute_error")[
            ["engine_id", "cycle", "true_rul",
             "predicted_rul", "error", "absolute_error"]
        ].to_string(index=False)
    )

    # Figure 1: Predictions versus ground truth.
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.65)
    limit = max(y_true.max(), y_pred.max())
    plt.plot([0, limit], [0, limit], "--")
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Random Forest: Actual vs Predicted RUL")
    plt.tight_layout()
    plt.savefig(FIGURES / "actual_vs_predicted.png", dpi=180)
    plt.close()

    # Figure 2: Residual distribution.
    plt.figure(figsize=(8, 5))
    plt.hist(df["error"], bins=18, edgecolor="black")
    plt.axvline(0, linestyle="--")
    plt.xlabel("Prediction Error (Cycles)")
    plt.ylabel("Number of Endpoints")
    plt.title("Random Forest: Error Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES / "error_distribution.png", dpi=180)
    plt.close()

    # Figure 3: Errors by RUL horizon.
    groups = [
        group["absolute_error"].to_numpy()
        for _, group in df.groupby("true_rul")
    ]

    labels = [
        str(int(horizon))
        for horizon in sorted(df["true_rul"].unique())
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(groups, tick_labels=labels)
    plt.xlabel("True RUL Horizon")
    plt.ylabel("Absolute Error (Cycles)")
    plt.title("Random Forest: Error by RUL Horizon")
    plt.tight_layout()
    plt.savefig(FIGURES / "error_by_horizon.png", dpi=180)
    plt.close()

    print("\nAnalysis figures saved to:", FIGURES)


if __name__ == "__main__":
    main()

