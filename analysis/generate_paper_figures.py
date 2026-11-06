"""Generate figures only from available evidence CSV files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate(evidence: Path, output: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Figure generation requires matplotlib and pandas") from exc
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    expected = [
        "clean_fraction_vs_tolerance",
        "force_error_vs_tolerance",
        "reuse_accuracy_frontier",
        "zeroth_vs_first_order",
        "error_distributions",
        "edge_error_vs_displacement",
        "edge_error_vs_reference_age",
        "generalization_seeds",
        "temperature_generalization",
        "system_size_scaling",
        "inference_runtime_vs_clean_fraction",
        "break_even_analysis",
        "rollout_runtime",
        "rollout_force_error",
    ]
    frontier = evidence / "frontier_summary.csv"
    if frontier.exists():
        frame = pd.read_csv(frontier)
        required = {"tolerance_A", "n_clean", "n_edges", "force_rms_error_meV_A"}
        if required.issubset(frame.columns):
            frame["clean_fraction"] = frame["n_clean"] / frame["n_edges"]
            plots = [
                ("reuse_accuracy_frontier", frame["clean_fraction"], frame["force_rms_error_meV_A"],
                 "clean-edge fraction", "force RMS error (meV/A)"),
                ("clean_fraction_vs_tolerance", frame["tolerance_A"], frame["clean_fraction"],
                 "tolerance (A)", "clean-edge fraction"),
                ("force_error_vs_tolerance", frame["tolerance_A"], frame["force_rms_error_meV_A"],
                 "tolerance (A)", "force RMS error (meV/A)"),
            ]
            for name, x, y, xlabel, ylabel in plots:
                figure, axis = plt.subplots(figsize=(6, 4))
                axis.scatter(x, y, s=10)
                axis.set(xlabel=xlabel, ylabel=ylabel)
                figure.tight_layout()
                figure.savefig(output / f"{name}.png", dpi=180)
                plt.close(figure)
                generated.append(f"{name}.png")
    (output / "figure_manifest.json").write_text(
        json.dumps({
            "expected_figures": expected,
            "generated_figures": generated,
            "missing_figures": [name for name in expected if f"{name}.png" not in generated],
            "missing_data_is_not_filled": True,
        }, indent=2),
        encoding="utf-8",
    )
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("results/active/paper_evidence"))
    parser.add_argument("--output", type=Path, default=Path("results/active/paper_evidence/figures"))
    args = parser.parse_args()
    print(generate(args.evidence, args.output))
