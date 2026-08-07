"""Aggregate comparable scalar metrics across seeds/configurations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads(path.read_text()) for path in args.inputs]
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        grouped.setdefault(run["model_type"], []).append(run)
    summary = {}
    excluded = {"sampling_diagnostics", "dataset_metadata", "parameter_counts", "generation_config", "sample_texts"}
    for model_type, model_runs in grouped.items():
        metrics = {}
        keys = set.intersection(*(set(run) for run in model_runs)) - excluded
        for key in sorted(keys):
            values = [run[key] for run in model_runs]
            if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                metrics[key] = {
                    "mean": statistics.fmean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "count": len(values),
                }
        summary[model_type] = metrics
    output = {"runs": len(runs), "groups": summary, "input_files": [str(path) for path in args.inputs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

