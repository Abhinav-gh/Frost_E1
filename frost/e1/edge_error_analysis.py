"""Analysis helpers for edge-level cached-message error studies.

Stage B currently records aggregate message errors.  This module deliberately
does not expand aggregate rows into fabricated per-edge observations.  It
accepts a per-edge CSV produced by an instrumented run and fails clearly when
the required columns are absent.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


REQUIRED_COLUMNS = {
    "edge_displacement_A",
    "reference_age_steps",
    "message_error",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No per-edge rows found in {path}")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(
            f"{path} is an aggregate result, not a per-edge result; "
            f"missing columns: {', '.join(sorted(missing))}"
        )
    return rows


def _number(rows: Iterable[dict[str, str]], column: str) -> np.ndarray:
    return np.asarray([float(row[column]) for row in rows], dtype=float)


def _summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def analyze(input_csv: Path, output_dir: Path, bins: int = 10) -> dict:
    """Write per-edge summaries and correlations without changing raw data."""
    if bins < 1:
        raise ValueError("bins must be positive")
    rows = _read_rows(input_csv)
    displacement = _number(rows, "edge_displacement_A")
    age = _number(rows, "reference_age_steps")
    error = _number(rows, "message_error")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_fields = list(rows[0])
    with (output_dir / "per_edge_message_error.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(rows)

    edges = np.linspace(float(displacement.min()), float(displacement.max()), bins + 1)
    grouped = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (displacement >= lower) & (
            displacement <= upper if upper == edges[-1] else displacement < upper
        )
        grouped.append({
            "displacement_lower_A": float(lower),
            "displacement_upper_A": float(upper),
            "message_error": _summary(error[mask]),
        })

    summary = {
        "input": str(input_csv),
        "n_edges": int(error.size),
        "message_error": _summary(error),
        "correlation": {
            "displacement_vs_message_error": _pearson(displacement, error),
            "reference_age_vs_message_error": _pearson(age, error),
        },
        "grouped_by_displacement": grouped,
        "interpretation": (
            "Correlations describe association only. They do not establish that "
            "displacement causes message error or that displacement is a sufficient predictor."
        ),
    }
    (output_dir / "edge_error_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (output_dir / "edge_error_by_displacement.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["displacement_lower_A", "displacement_upper_A", "n", "mean", "median", "p95", "max"],
        )
        writer.writeheader()
        for item in grouped:
            stats = item["message_error"]
            writer.writerow({
                "displacement_lower_A": item["displacement_lower_A"],
                "displacement_upper_A": item["displacement_upper_A"],
                **{key: stats.get(key, "") for key in ("n", "mean", "median", "p95", "max")},
            })
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/active/edge_error"))
    parser.add_argument("--bins", type=int, default=10)
    options = parser.parse_args()
    print(json.dumps(analyze(options.input, options.output_dir, options.bins), indent=2))
