"""Turbofan Predictive Maintenance — RUL Estimation (Streamlit demo).

Run with:
    streamlit run demo/app.py

The app only offers models whose checkpoints exist on disk. Every
prediction goes through the same preprocessing and inference code used
by the official evaluation, so the demo cannot produce numbers that the
report does not recognise.
"""

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.windowing import create_last_windows  # noqa: E402
from src.experiments.registry import read_registry  # noqa: E402
from src.inference.pipeline import (  # noqa: E402
    FEATURE_COLUMNS,
    InputValidationError,
    load_artifacts,
    load_sample_engine,
    prepare_scaled_frame,
    validate_raw_frame,
)
from src.inference.predictor import RULPredictor  # noqa: E402
from src.inference.demo_state import (  # noqa: E402
    input_signature,
    is_official_reference_allowed,
)


PROCESSED_DIR = ROOT / "data" / "processed"
TEST_RUL_PATH = PROCESSED_DIR / "fd001_test_rul.csv"
CHECKPOINT_DIR = ROOT / "checkpoints"

MODEL_DESCRIPTIONS = {
    "random_forest": (
        "Random Forest on statistical window features (last, mean, std, change). Baseline."
    ),
    "optimized_random_forest": (
        "Random Forest hyperparameter study winner "
        "(n_estimators=150, max_features='sqrt', "
        "min_samples_leaf=1, max_depth=20)."
    ),
    "lstm": "Stacked LSTM (2 layers, 64 hidden units), packed sequences.",
    "transformer": (
        "Transformer encoder (2 layers, 4 heads, d_model=64) with "
        "padding-aware attention and masked mean pooling."
    ),
    "cnn_bilstm_attention": (
        "CNN + BiLSTM + masked additive attention over time."
    ),
    "cnn_bilstm_mean": (
        "Attention ablation: identical CNN + BiLSTM with masked mean "
        "pooling instead of attention."
    ),
}

st.set_page_config(
    page_title="Turbofan Predictive Maintenance — RUL Estimation",
    page_icon="🛠️",
    layout="wide",
)


@st.cache_data(show_spinner=False, ttl=300)
def cached_metadata():
    metadata, scaler = load_artifacts()
    return metadata, scaler


@st.cache_data(show_spinner=False, ttl=30)
def available_checkpoints() -> dict[str, str]:
    """Map model label -> checkpoint path for models that really exist."""

    found: dict[str, str] = {}

    rf_path = CHECKPOINT_DIR / "random_forest.joblib"

    if rf_path.is_file():
        found["random_forest"] = str(rf_path)

    # Optimized Random Forest from the hyper-parameter study. Its
    # checkpoint stores the fitted estimator directly (no dict wrapper);
    # RULPredictor handles both formats.
    optimized_rf_path = CHECKPOINT_DIR / "random_forest_optimized.joblib"

    if optimized_rf_path.is_file():
        found["optimized_random_forest"] = str(optimized_rf_path)

    registry = read_registry()

    if registry.empty:
        return found

    main_runs = registry[
        (registry["tag"] == "main")
        & (registry["status"] == "completed")
    ]

    for model, group in main_runs.groupby("model"):
        best = group.sort_values("val_rmse").iloc[0]
        checkpoint = ROOT / str(best["checkpoint"])

        if checkpoint.is_file():
            found[str(model)] = str(checkpoint)

    ablation = registry[
        (registry["tag"] == "ablation")
        & (registry["status"] == "completed")
    ]

    if not ablation.empty:
        best = ablation.sort_values("val_rmse").iloc[0]
        checkpoint = ROOT / str(best["checkpoint"])

        if checkpoint.is_file():
            found["cnn_bilstm_mean"] = str(checkpoint)

    return found


@st.cache_resource(show_spinner=False)
def load_predictor(path: str) -> RULPredictor:
    """Load a checkpoint; RULPredictor handles joblib and PyTorch files."""
    return RULPredictor(Path(path), device="cpu")


def predict_frame(df: pd.DataFrame, predictor: RULPredictor, metadata, scaler):
    scaled = prepare_scaled_frame(df, metadata, scaler)

    windows = create_last_windows(
        scaled,
        features=metadata["selected_features"],
        window_size=predictor.window_size,
    )

    predictions = predictor.predict_windows(
        windows["X"],
        windows["mask"],
    )

    return pd.DataFrame({
        "engine_id": windows["engines"],
        "last_cycle": windows["cycles"],
        "window_length": windows["lengths"],
        "predicted_rul": predictions,
    })


def reference_labels() -> pd.DataFrame:
    if not TEST_RUL_PATH.is_file():
        return pd.DataFrame()

    return pd.read_csv(TEST_RUL_PATH)


def page_home(checkpoints: dict[str, str], metadata: dict):
    st.title("Turbofan Predictive Maintenance — RUL Estimation")

    st.markdown(
        """
### What is RUL?

**Remaining Useful Life (RUL)** is the number of operating cycles an
engine has left before it fails. Predicting it from sensor history is
the core task of predictive maintenance: too late and the engine fails
in service, too early and healthy parts are replaced for nothing.
"""
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Dataset", "NASA C-MAPSS FD001")
    col_b.metric("Selected sensors", f"{len(metadata['selected_features'])}")
    col_c.metric("Available models", f"{len(checkpoints)}")

    st.markdown(
        f"""
### Dataset

* **Source:** NASA Prognostics Center of Excellence (PCoE), C-MAPSS
  Turbofan Engine Degradation Simulation Data Set.
* **Sub-set:** FD001 — one operating condition, one fault mode.
* **Training engines:** {metadata['train_engines']} run-to-failure
  trajectories ({metadata['train_rows']} cycles).
* **Validation engines:** {metadata['validation_engines']}
  ({metadata['validation_rows']} cycles), engine-wise split.
* **Official test engines:** {metadata['test_engines']} truncated
  trajectories, labels from `RUL_FD001.txt`.

### Models

| Model | Idea |
| --- | --- |
| Random Forest | baseline on statistical window features |
| LSTM | recurrent sequence model with packed sequences |
| Transformer | self-attention over the window with padding masks |
| CNN-BiLSTM-Attention | convolutional features + bidirectional LSTM + additive attention |
"""
    )

    st.info(
        "Every model shown in this app has a checkpoint on disk. "
        "Models without a checkpoint are not listed."
    )


def input_controls(metadata, scaler):
    """Render the input picker inside the sidebar and return a frame."""

    with st.sidebar:
        st.markdown("### Input")

        mode = st.radio(
            "Source",
            [
                "Sample engine from the dataset",
                "Upload a sensor CSV",
            ],
            key="input_source",
        )

        if mode == "Sample engine from the dataset":
            engines = list(range(1, metadata["test_engines"] + 1))

            engine_id = st.selectbox(
                "Test engine",
                engines,
                index=0,
                help=(
                    "Official test engines are truncated trajectories, "
                    "so the model has to predict the remaining life."
                ),
            )

            last_cycles = st.slider(
                "Cycles to load",
                min_value=10,
                max_value=200,
                value=60,
                step=5,
            )

            try:
                df = load_sample_engine(
                    engine_id, "test", last_cycles
                )
            except (FileNotFoundError, InputValidationError) as error:
                st.error(str(error))
                return None

            st.caption(
                f"Test engine {engine_id}: {len(df)} cycles loaded."
            )

            return df

        uploaded = st.file_uploader(
            "Sensor CSV (26 C-MAPSS columns)",
            type=["csv"],
            help=(
                "engine_id, cycle, setting_1..setting_3, "
                "sensor_1..sensor_21"
            ),
        )

        if uploaded is None:
            st.caption(
                "`engine_id` and `cycle` are optional and filled "
                "automatically when missing."
            )
            return None

        try:
            df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            df = validate_raw_frame(df)
            if df["engine_id"].nunique() != 1:
                raise InputValidationError(
                    "Upload one engine history per CSV for the live demo."
                )
        except InputValidationError as error:
            st.error(str(error))
            return None
        except Exception as error:  # noqa: BLE001
            st.error(f"Could not read the CSV file: {error}")
            return None

        st.success(f"Valid sensor data: {len(df)} rows.")

        return df


def page_input_summary(df):
    st.header("Input")

    if df is None:
        st.info(
            "Pick a sample engine or upload a CSV from the sidebar to "
            "start."
        )
        return

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Engine ID", int(df["engine_id"].iloc[0]))
    col_b.metric("Cycles", int(df["cycle"].max()))
    col_c.metric("Rows", len(df))

    st.subheader("Validated input preview")
    st.dataframe(df.head(12), width="stretch")

    st.caption(
        "Validation performed: required columns, numeric values, "
        "positive ids, unique (engine_id, cycle), consecutive cycles."
    )

    csv_bytes = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download this input as CSV",
        data=csv_bytes,
        file_name="rul_input.csv",
        mime="text/csv",
    )


def page_monitoring(df: pd.DataFrame):
    st.header("Engine monitoring")

    if df is None:
        st.info("Load an input first.")
        return

    engine_id = int(df["engine_id"].iloc[0])

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Engine ID", engine_id)
    col_b.metric("Available cycles", int(df["cycle"].max()))
    col_c.metric("Rows", len(df))

    settings = [c for c in df.columns if c.startswith("setting_")]

    if settings:
        st.caption("Operating conditions (mean over the loaded window)")
        st.dataframe(
            df[settings].describe().loc[["mean", "std", "min", "max"]],
            width="stretch",
        )

    sensors = [
        c for c in FEATURE_COLUMNS
        if c.startswith("sensor_") and c in df.columns
    ]

    default = [s for s in ["sensor_2", "sensor_3", "sensor_4", "sensor_11"] if s in sensors]

    selected = st.multiselect(
        "Sensors to plot",
        sensors,
        default=default or sensors[:3],
    )

    if selected:
        fig, axes = plt.subplots(
            len(selected), 1,
            figsize=(9, 2.2 * len(selected)),
            sharex=True,
            squeeze=False,
        )

        for ax, sensor in zip(axes[:, 0], selected):
            ax.plot(df["cycle"], df[sensor], linewidth=1.2)
            ax.set_ylabel(sensor, fontsize=8)
            ax.grid(alpha=0.3)

        axes[-1, 0].set_xlabel("Cycle")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)


def page_prediction(df, metadata, scaler, checkpoints):
    st.header("Prediction")

    if df is None:
        st.info("Load an input first.")
        return

    if not checkpoints:
        st.error(
            "No trained checkpoints found. Train a model first, then "
            "reload this page."
        )
        return

    labels = list(checkpoints)

    choice = st.selectbox(
        "Model",
        labels,
        format_func=lambda name: f"{name} — {MODEL_DESCRIPTIONS.get(name, '')}",
    )

    # A stored result is valid only for the exact input, source and model.
    signature = input_signature(
        df,
        model_name=choice,
        source=st.session_state.get("input_source", ""),
    )
    if st.session_state.get("prediction_signature") != signature:
        st.session_state.pop("results", None)
        st.session_state.pop("model_used", None)

    if st.button("Estimate RUL", type="primary"):
        with st.spinner("Running inference..."):
            try:
                predictor = load_predictor(checkpoints[choice])
                results = predict_frame(df, predictor, metadata, scaler)
            except InputValidationError as error:
                st.error(str(error))
                return
            except FileNotFoundError as error:
                st.error(str(error))
                return
            except Exception as error:  # noqa: BLE001
                st.error(f"Inference failed: {error}")
                return

        st.session_state["results"] = results
        st.session_state["model_used"] = choice
        st.session_state["prediction_signature"] = signature

    results = st.session_state.get("results")

    if (
        results is None
        or st.session_state.get("model_used") != choice
        or st.session_state.get("prediction_signature") != signature
    ):
        return

    prediction = float(results["predicted_rul"].iloc[0])

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Estimated RUL", f"{prediction:.1f} cycles")
    col_b.metric("Model", choice)
    col_c.metric("Cycles used", int(results["window_length"].iloc[0]))

    # Official labels belong exclusively to the bundled NASA test sample.
    # A user-uploaded CSV may reuse an engine ID but has no known RUL label.
    if is_official_reference_allowed(
        source=st.session_state.get("input_source", ""),
        input_frame=df,
        predictions=results,
    ):
        reference = reference_labels()
        if not reference.empty:
            engine_id = int(results["engine_id"].iloc[0])
            row = reference[reference["engine_id"] == engine_id]
            if not row.empty:
                truth = float(row["rul"].iloc[0])
                st.caption(
                    f"Official reference label for test engine {engine_id}: "
                    f"{truth:.0f} cycles (error "
                    f"{prediction - truth:+.1f})."
                )

    st.write("Input summary")
    st.dataframe(
        results.rename(columns={"predicted_rul": "predicted_rul (cycles)"}),
        width="stretch",
    )

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.bar(["Estimated"], [prediction], color="#4C72B0")
    ax.axhline(prediction, linestyle="--", color="grey", linewidth=1)
    ax.set_ylabel("RUL (cycles)")
    ax.set_title("Predicted remaining useful life")
    ax.set_ylim(0, max(150, prediction * 1.2))
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def page_comparison(df, metadata, scaler, checkpoints):
    st.header("Model comparison")

    if df is None:
        st.info("Load an input first.")
        return

    if len(checkpoints) < 2:
        st.info("At least two trained models are needed for a comparison.")
        return

    if not st.button("Compare all available models"):
        return

    rows = []

    with st.spinner("Running every available model..."):
        for label, path in checkpoints.items():
            try:
                predictor = load_predictor(path)
                results = predict_frame(df, predictor, metadata, scaler)
                rows.append({
                    "model": label,
                    "predicted_rul": float(
                        results["predicted_rul"].iloc[0]
                    ),
                })
            except Exception as error:  # noqa: BLE001
                rows.append({
                    "model": label,
                    "predicted_rul": np.nan,
                    "error": str(error),
                })

    table = pd.DataFrame(rows)

    st.dataframe(table, width="stretch")

    valid = table.dropna(subset=["predicted_rul"])

    if not valid.empty:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.barh(
            valid["model"],
            valid["predicted_rul"],
            color="#55A868",
        )
        ax.set_xlabel("Predicted RUL (cycles)")
        ax.set_title("RUL estimate per model for the same input")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.caption(
        "Different models are trained on the same split; disagreements "
        "here reflect model variance, not different inputs."
    )


def page_explainability(df, metadata, checkpoints):
    st.header("Explainability")

    if df is None:
        st.info("Load an input first.")
        return

    st.markdown(
        """
The plots below show how the sensors behave over the loaded window.
They describe **correlation with degradation**, not causation: no
attention weight or trend is proof that a specific sensor caused the
failure.
"""
    )

    sensors = [
        c for c in FEATURE_COLUMNS
        if c.startswith("sensor_") and c in df.columns
    ]

    if not sensors:
        return

    fig, ax = plt.subplots(figsize=(9, 3.6))

    for sensor in sensors[:6]:
        values = df[sensor].to_numpy(dtype=float)
        span = values.std() or 1.0
        ax.plot(
            df["cycle"],
            (values - values.mean()) / span,
            label=sensor,
            linewidth=1.1,
        )

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Standardized sensor reading")
    ax.set_title("Sensor trends in the loaded window")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    first_last = pd.DataFrame({
        "sensor": sensors,
        "first": [float(df[s].iloc[0]) for s in sensors],
        "last": [float(df[s].iloc[-1]) for s in sensors],
    })

    first_last["relative_change"] = (
        first_last["last"] - first_last["first"]
    ) / first_last["first"].abs().replace(0, np.nan)

    st.write("Largest relative changes across the window")
    st.dataframe(
        first_last.reindex(
            first_last["relative_change"].abs()
            .sort_values(ascending=False).index
        ).head(8),
        width="stretch",
    )


def main():
    try:
        metadata, scaler = cached_metadata()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    checkpoints = available_checkpoints()

    pages = [
        "Home",
        "Input",
        "Monitoring",
        "Prediction",
        "Comparison",
        "Explainability",
    ]

    with st.sidebar:
        st.title("🛠️ RUL Estimation")
        page = st.radio("Navigation", pages)

        st.markdown("---")
        st.caption("Available models")

        if checkpoints:
            for label in checkpoints:
                st.markdown(f"- `{label}`")
        else:
            st.warning("No checkpoints found yet.")

        st.markdown("---")
        st.caption(
            "Dataset: NASA C-MAPSS FD001. "
            "Predictions are research output, not a maintenance authority."
        )

    if page == "Home":
        page_home(checkpoints, metadata)
        return

    df = input_controls(metadata, scaler)

    # Never reuse a previous input when a file is removed or the source changes.
    active_df = df
    if active_df is None:
        st.session_state.pop("results", None)
        st.session_state.pop("model_used", None)
        st.session_state.pop("prediction_signature", None)

    if page == "Input":
        page_input_summary(active_df)
        return

    if page == "Monitoring":
        page_monitoring(active_df)
    elif page == "Prediction":
        page_prediction(active_df, metadata, scaler, checkpoints)
    elif page == "Comparison":
        page_comparison(active_df, metadata, scaler, checkpoints)
    elif page == "Explainability":
        page_explainability(active_df, metadata, checkpoints)


if __name__ == "__main__":
    main()
