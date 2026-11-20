"""Exploratory data analysis figures for FD001.

Run with:
    python -m src.datasets.visualize
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "train_FD001.txt"
OUTPUT_DIR = ROOT / "experiments" / "figures" / "eda"

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Raw FD001 training data not found: {DATA_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    # RUL labels for training trajectories only.
    max_cycles = df.groupby("engine_id")["cycle"].transform("max")
    df["rul"] = max_cycles - df["cycle"]

    # Figure 1: engine lifetime distribution.
    lifetimes = df.groupby("engine_id")["cycle"].max()

    plt.figure(figsize=(9, 5))
    plt.hist(lifetimes, bins=15, edgecolor="black")
    plt.xlabel("Engine Lifetime (Cycles)")
    plt.ylabel("Number of Engines")
    plt.title("FD001 — Engine Lifetime Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "engine_lifetimes.png", dpi=180)
    plt.close()

    # Figure 2: sensor trends across selected engines.
    engine_ids = [1, 25, 50]
    sensors = [
        "sensor_2", "sensor_3", "sensor_4",
        "sensor_9", "sensor_11", "sensor_15",
    ]

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))

    for ax, sensor in zip(axes.flatten(), sensors):
        for engine_id in engine_ids:
            subset = df[df["engine_id"] == engine_id]
            ax.plot(
                subset["cycle"],
                subset[sensor],
                label=f"Engine {engine_id}",
                linewidth=1.3,
            )

        ax.set_title(sensor)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Sensor Reading")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sensor_trends.png", dpi=180)
    plt.close()

    # Figure 3: correlation with training RUL.
    sensors = [f"sensor_{i}" for i in range(1, 22)]

    correlation = (
        df[sensors + ["rul"]]
        .corr()["rul"]
        .drop("rul")
        .dropna()
        .sort_values()
    )

    plt.figure(figsize=(10, 7))
    correlation.plot(kind="barh")
    plt.xlabel("Pearson Correlation with RUL")
    plt.title("FD001 — Sensor Correlation with RUL")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "sensor_rul_correlation.png",
        dpi=180,
    )
    plt.close()

    print(f"EDA figures saved to: {OUTPUT_DIR}")
    print("\nSensor-RUL correlations:")
    print(correlation.round(3))


if __name__ == "__main__":
    main()
