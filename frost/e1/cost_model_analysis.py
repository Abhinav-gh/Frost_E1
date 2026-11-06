"""Measured-versus-predicted Frost cost model utilities."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def predict_runtime(
    baseline_seconds: float,
    targeted_seconds_saved: float,
    dirty_seconds: float,
    derivative_seconds: float,
    geometry_seconds: float,
    cache_seconds: float,
    synchronization_seconds: float = 0.0,
) -> float:
    """Evaluate the documented additive decomposition of cached runtime."""
    values = (
        baseline_seconds, targeted_seconds_saved, dirty_seconds,
        derivative_seconds, geometry_seconds, cache_seconds,
        synchronization_seconds,
    )
    if any(value < 0 for value in values):
        raise ValueError("cost components must be non-negative")
    if targeted_seconds_saved > baseline_seconds:
        raise ValueError("targeted work saved cannot exceed baseline runtime")
    return (
        baseline_seconds
        - targeted_seconds_saved
        + dirty_seconds
        + derivative_seconds
        + geometry_seconds
        + cache_seconds
        + synchronization_seconds
    )


def analyze(input_csv: Path, output_csv: Path) -> list[dict[str, float | str]]:
    """Compute predicted and measured speedups from a component CSV.

    Required columns are explicit so a missing measurement cannot silently
    become zero and produce a success-shaped result.
    """
    required = {
        "arm", "baseline_seconds", "targeted_seconds_saved",
        "dirty_seconds", "derivative_seconds", "geometry_seconds",
        "cache_seconds", "synchronization_seconds", "measured_seconds",
    }
    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {input_csv}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing cost-model columns: {', '.join(sorted(missing))}")

    result = []
    for row in rows:
        numbers = {key: float(row[key]) for key in required if key != "arm"}
        predicted = predict_runtime(**{
            key: numbers[key] for key in (
                "baseline_seconds", "targeted_seconds_saved", "dirty_seconds",
                "derivative_seconds", "geometry_seconds", "cache_seconds",
                "synchronization_seconds",
            )
        })
        measured = numbers["measured_seconds"]
        result.append({
            "arm": row["arm"],
            "predicted_seconds": predicted,
            "measured_seconds": measured,
            "prediction_residual_seconds": measured - predicted,
            "predicted_speedup": numbers["baseline_seconds"] / predicted if predicted else "",
            "measured_speedup": numbers["baseline_seconds"] / measured if measured else "",
        })
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/active/cost_model.csv"))
    options = parser.parse_args()
    print(json.dumps(analyze(options.input, options.output), indent=2))
