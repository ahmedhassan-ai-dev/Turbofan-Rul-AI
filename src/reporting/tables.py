"""Small markdown helpers so reports never depend on `tabulate`."""

from pathlib import Path

import pandas as pd


def _format(value) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        return f"{value:.4f}".rstrip("0").rstrip(".")

    return str(value)


def to_markdown(df: pd.DataFrame) -> str:
    """Render a dataframe as a GitHub-flavoured markdown table."""

    if df.empty:
        return "_No rows._"

    columns = [str(column) for column in df.columns]

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for row in df.itertuples(index=False):
        cells = [_format(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def write_markdown(df: pd.DataFrame, path: Path, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {title}\n\n{to_markdown(df)}\n",
        encoding="utf-8",
    )
