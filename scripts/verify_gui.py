"""Headless end-to-end test of the Streamlit GUI.

Run from the repository root::

    python scripts/verify_gui.py

Uses ``streamlit.testing.v1.AppTest``, which executes ``demo/app.py`` with
the real widget tree (no browser needed): navigation, model selection,
the Predict button and the Comparison page are all driven the same way a
user would drive them.
"""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "demo" / "app.py"

MODELS = [
    "random_forest",
    "optimized_random_forest",
    "lstm",
    "transformer",
    "cnn_bilstm_attention",
]


def find(app, kind: str, label: str):
    for widget in app.get(kind):
        if getattr(widget, "label", None) == label:
            return widget
    raise AssertionError(f"{kind} with label {label!r} not found")


def errors(app) -> list[str]:
    return [element.value for element in app.error]


def state(app, key, default=None):
    """SafeSessionState has no .get(); read defensively."""
    try:
        return app.session_state[key]
    except KeyError:
        return default


def main() -> int:
    failures: list[str] = []

    app = AppTest.from_file(str(APP), default_timeout=300)
    app.run()

    if app.exception:
        print("APP FAILED TO START:", app.exception)
        return 1

    print("=" * 74)
    print("GUI STARTUP")
    print("=" * 74)
    print(f"page: {APP}")
    print(f"error elements on load: {errors(app) or 'none'}")

    nav = app.sidebar.radio[0]
    print(f"navigation options: {nav.options}")

    # ------------------------------------------------------------------
    # sidebar lists every checkpoint
    # ------------------------------------------------------------------
    print()
    print("=" * 74)
    print("CHECKPOINT DISCOVERY (sidebar)")
    print("=" * 74)
    sidebar_text = " ".join(
        element.value
        for element in list(app.sidebar.markdown) + list(app.sidebar.caption)
    )
    for model in MODELS:
        present = model in sidebar_text
        print(f"  {model:26s} listed={present}")
        if not present:
            failures.append(f"{model} missing from sidebar model list")

    # ------------------------------------------------------------------
    # Prediction page, one model at a time
    # ------------------------------------------------------------------
    print()
    print("=" * 74)
    print("PREDICTION PAGE")
    print("=" * 74)

    nav.set_value("Prediction").run()

    source = find(app, "radio", "Source")
    print(f"input source: {source.value!r} (sample engine {find(app, 'selectbox', 'Test engine').value})")

    for model in MODELS:
        select = find(app, "selectbox", "Model")

        # options are rendered with format_func: "<name> — <description>"
        labels = {str(option).split(" — ")[0]: option for option in select.options}

        if model not in labels:
            failures.append(f"{model}: not offered by the Model selectbox")
            print(f"{model:26s} NOT OFFERED")
            continue

        # set_value takes the raw option value, not the formatted label
        select.set_value(model).run()

        button = find(app, "button", "Estimate RUL")
        button.click().run()

        if app.exception:
            failures.append(f"{model}: uncaught exception {app.exception[0].value}")
            print(f"{model:26s} EXCEPTION {app.exception[0].value}")
            continue

        found_errors = errors(app)
        results = state(app, "results")
        used = state(app, "model_used")

        if found_errors:
            failures.append(f"{model}: {found_errors[0]}")
            print(f"{model:26s} ERROR: {found_errors[0]}")
            continue

        if results is None or used != model:
            failures.append(f"{model}: no results stored")
            print(f"{model:26s} NO RESULT")
            continue

        value = float(results["predicted_rul"].iloc[0])
        valid = value == value and value >= 0

        print(f"{model:26s} RUL = {value:8.3f} cycles   {'OK' if valid else 'FAIL'}")

        if not valid:
            failures.append(f"{model}: invalid prediction {value}")

    # ------------------------------------------------------------------
    # Comparison page
    # ------------------------------------------------------------------
    print()
    print("=" * 74)
    print("MODEL COMPARISON PAGE")
    print("=" * 74)

    nav.set_value("Comparison").run()

    compare = find(app, "button", "Compare all available models")
    compare.click().run()

    tables = app.get("dataframe")
    if app.exception or not tables:
        failures.append(f"comparison page failed: {app.exception or 'no table'}")
        print("comparison FAILED:", app.exception)
    else:
        table = tables[0].value
        print(table.to_string(index=False))
        compared = set(table["model"])
        missing = set(MODELS) - compared
        if missing:
            failures.append(f"comparison missing models: {sorted(missing)}")
        nan_rows = table[table["predicted_rul"].isna()]
        if len(nan_rows):
            failures.append(f"comparison has failed predictions: {nan_rows.to_dict('records')}")

    print()
    if failures:
        print("FAILURES:")
        for failure in failures:
            print(" -", failure)
        return 1

    print("GUI VERIFICATION PASSED: 5/5 models predicted, comparison OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
