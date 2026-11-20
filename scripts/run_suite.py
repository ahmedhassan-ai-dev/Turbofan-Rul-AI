"""Run a suite of experiments from a JSON file, resumably.

Usage
-----
    python scripts/run_suite.py configs/experiments/main.json
    python scripts/run_suite.py configs/experiments/main.json --dry-run
    python scripts/run_suite.py configs/experiments/main.json --force

A configuration whose run directory already contains a valid
``summary.json`` is skipped, so an interrupted sweep can be resumed
without retraining finished models.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.registry import read_registry  # noqa: E402
from src.experiments.runner import run_experiment  # noqa: E402


def is_complete(experiment_id: str) -> bool:
    run_dir = ROOT / "experiments" / "runs" / experiment_id

    return (
        (run_dir / "summary.json").is_file()
        and (run_dir / "best_model.pt").is_file()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    suite_path = Path(args.suite)

    if not suite_path.is_absolute():
        suite_path = ROOT / suite_path

    suite = json.loads(suite_path.read_text(encoding="utf-8"))

    defaults = {
        key: value
        for key, value in suite.get("defaults", {}).items()
        if value is not None
    }

    # Suite defaults fill anything a run does not set explicitly.
    runs = [{**defaults, **run} for run in suite["runs"]]

    print(f"=== SUITE: {suite.get('name', suite_path.stem)} ===")
    print(f"Experiments: {len(runs)}")

    executed = 0
    skipped = 0

    for index, config in enumerate(runs, start=1):
        experiment_id = config["experiment_id"]

        if is_complete(experiment_id) and not args.force:
            print(f"[{index}/{len(runs)}] SKIP {experiment_id} (exists)")
            skipped += 1
            continue

        if args.dry_run:
            print(f"[{index}/{len(runs)}] WOULD RUN {experiment_id}")
            continue

        if args.limit is not None and executed >= args.limit:
            print("Reached --limit, stopping.")
            break

        print(f"\n[{index}/{len(runs)}] RUN {experiment_id}")

        start = time.perf_counter()

        try:
            summary = run_experiment(config)
        except RuntimeError as error:
            print(f"FAILED {experiment_id}: {error}")
            raise

        duration = time.perf_counter() - start

        print(
            f"DONE {experiment_id} in {duration / 60:.1f} min | "
            f"val RMSE {summary['best_val_rmse']:.3f}"
        )

        executed += 1

    print(f"\nExecuted: {executed} | Skipped: {skipped}")

    registry = read_registry()

    if not registry.empty:
        print("\n=== REGISTRY (validation) ===")
        print(
            registry[
                [
                    "experiment_id", "tag", "model", "seed",
                    "window_size", "rul_cap", "val_rmse",
                    "best_epoch", "status",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
