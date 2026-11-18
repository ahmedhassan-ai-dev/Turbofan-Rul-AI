
# ============================================================
# STEP 40.1 | LIVE RUL PREDICTION DEMO
# NASA C-MAPSS FD001 | Gradio
# Frozen checkpoints | No training | CPU-compatible
# ============================================================

import json
import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.datasets.preprocessing import COLUMNS, validate_dataset
from src.datasets.windowing import create_windows
from src.train import build_model


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

RUNS = Path(
    os.environ.get(
        "TURBOFAN_RUNS_DIR",
        str(ROOT / "experiments" / "runs"),
    )
)

DEFAULT_DATA = ROOT / "data" / "raw" / "test_FD001.txt"

MODEL_CONFIGS = {
    "Transformer": (
        "transformer",
        "transformer_cap125_lr3e4_seed{seed}",
    ),
    "LSTM": (
        "lstm",
        "lstm_v2_cap125_seed{seed}",
    ),
    "Hybrid": (
        "cnn_bilstm_attention",
        "hybrid_cap125_seed{seed}",
    ),
}


@lru_cache(maxsize=1)
def load_preprocessing():
    metadata_path = PROCESSED / "fd001_metadata.json"

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    features = metadata["selected_features"]

    scaler_path = PROCESSED / metadata["scaler_file"]
    scaler = joblib.load(scaler_path)

    assert len(features) == 17
    assert scaler.n_features_in_ == 17

    return features, scaler


@lru_cache(maxsize=9)
def load_model(model_choice, seed):
    if model_choice not in MODEL_CONFIGS:
        raise ValueError("Unsupported model")

    if seed not in (42, 123, 3407):
        raise ValueError("Unsupported experiment seed")

    model_name, folder_pattern = MODEL_CONFIGS[model_choice]

    checkpoint_path = (
        RUNS
        / folder_pattern.format(seed=seed)
        / "best_model.pt"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if float(checkpoint["rul_cap"]) != 125:
        raise ValueError("Unexpected checkpoint RUL cap")

    model = build_model(
        model_name=model_name,
        input_size=17,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()
    return model


def read_sensor_file(file_path):
    path = (
        Path(file_path)
        if file_path is not None
        else DEFAULT_DATA
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Sensor data file not found: {path}"
        )

    # Expected input: original NASA FD001 whitespace format.
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
    )

    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Expected {len(COLUMNS)} columns, "
            f"received {df.shape[1]}. "
            "Upload a raw C-MAPSS FD001-style TXT file."
        )

    df.columns = COLUMNS

    validate_dataset(df)
    return df


def predict_engine(
    file_path,
    engine_id,
    model_choice,
    seed,
):
    if engine_id is None or int(engine_id) != engine_id:
        raise ValueError("Engine ID must be an integer")

    engine_id = int(engine_id)
    seed = int(seed)

    raw = read_sensor_file(file_path)

    engine = raw.loc[
        raw["engine_id"] == engine_id
    ].sort_values("cycle").copy()

    if engine.empty:
        raise ValueError(
            f"Engine {engine_id} not found in the input file"
        )

    features, scaler = load_preprocessing()

    # Always transform using the original training scaler.
    scaled = engine.copy()
    scaled = scaled.astype({
        feature: "float64"
        for feature in features
    })

    scaled[features] = scaler.transform(
        scaled[features]
    )

    # create_windows() requires a target column.
    # This placeholder is never used to compute predictions.
    scaled["rul"] = 0.0

    windows = create_windows(
        df=scaled,
        features=features,
        window_size=30,
    )

    # Last observed window only — no future sensor readings.
    x = torch.from_numpy(
        windows["X"][-1:].copy()
    ).float()

    mask = torch.from_numpy(
        windows["mask"][-1:].copy()
    ).bool()

    model = load_model(model_choice, seed)

    with torch.inference_mode():
        predicted_rul = (
            model(x, mask)
            .clamp(min=0)
            .item()
        )

    # Visualization uses raw sensor measurements.
    recent = engine.tail(30)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(
        recent["cycle"],
        recent["sensor_11"],
        marker="o",
        markersize=3,
    )
    ax.set_title("Sensor 11 — Recent Observations")
    ax.set_xlabel("Operating Cycle")
    ax.set_ylabel("Raw Sensor Reading")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    details = (
        f"**Engine:** {engine_id}  \n"
        f"**Last observed cycle:** "
        f"{int(engine['cycle'].max())}  \n"
        f"**Observed history:** {len(engine)} cycles  \n"
        f"**Model:** {model_choice}  \n"
        f"**Checkpoint seed:** {seed}  \n"
        f"**Inference window:** 30 cycles  \n\n"
        "**Note:** Prediction is a research estimate "
        "from simulated C-MAPSS data, not a certified "
        "aircraft maintenance decision."
    )

    return round(predicted_rul, 2), details, fig


def run_prediction(file_path, engine_id, model_choice, seed):
    try:
        return predict_engine(
            file_path, engine_id, model_choice, seed
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


with gr.Blocks(
    title="Turbofan RUL Prediction"
) as demo:

    gr.Markdown(
        "# Turbofan Engine — Remaining Useful Life\n"
        "### NASA C-MAPSS FD001 | Deep Learning Demo\n\n"
        "Predict remaining operating cycles from sensor history "
        "using frozen models trained in PyTorch."
    )

    with gr.Row():
        with gr.Column():
            data_file = gr.File(
                label="Upload raw FD001 sensor data (optional)",
                file_types=[".txt"],
                type="filepath",
            )

            gr.Markdown(
                "Leave the file empty to use the included "
                "`test_FD001.txt` sample."
            )

            engine_input = gr.Number(
                label="Engine ID",
                value=1,
                precision=0,
            )

            model_input = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value="Transformer",
                label="Architecture",
            )

            seed_input = gr.Dropdown(
                choices=[42, 123, 3407],
                value=42,
                label="Checkpoint Seed",
            )

            predict_button = gr.Button(
                "Predict Remaining Useful Life",
                variant="primary",
            )

        with gr.Column():
            prediction_output = gr.Number(
                label="Predicted RUL (Cycles)"
            )

            details_output = gr.Markdown()

    sensor_plot = gr.Plot(
        label="Recent Sensor Trend"
    )

    predict_button.click(
        fn=run_prediction,
        inputs=[
            data_file,
            engine_input,
            model_input,
            seed_input,
        ],
        outputs=[
            prediction_output,
            details_output,
            sensor_plot,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        share=os.environ.get("GRADIO_SHARE") == "1",
    )

