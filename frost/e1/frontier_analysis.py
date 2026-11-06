"""Summarize reuse/error frontier with block-bootstrap uncertainty."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np


def analyze(input_csv: Path, output: Path, block: int, draws: int, seed: int):
    if block < 1 or draws < 1:
        raise ValueError("block length and bootstrap draws must be positive")
    rows = list(csv.DictReader(input_csv.open(newline="")))
    if not rows:
        raise ValueError(f"No rows in {input_csv}")
    groups = {}
    for row in rows:
        groups.setdefault(float(row["tolerance_A"]), []).append(row)
    rng = np.random.default_rng(seed)
    summary = []
    for tolerance, items in sorted(groups.items()):
        items.sort(key=lambda r: int(r["step"]))
        reuse = np.asarray([1.0-float(r.get("phi", 0)) for r in items])
        error = np.asarray([float(r["force_rms_error_meV_A"]) for r in items])
        def boot(values):
            blocks = [values[i:i+block] for i in range(0, len(values), block)]
            samples = np.empty(draws)
            for j in range(draws):
                selected = rng.integers(0, len(blocks), size=len(blocks))
                samples[j] = np.mean(np.concatenate([blocks[k] for k in selected]))
            return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
        summary.append({"tolerance_A": tolerance, "n_snapshots": len(items),
            "reuse_mean": float(reuse.mean()), "reuse_median": float(np.median(reuse)),
            "reuse_p95": float(np.percentile(reuse, 95)), "reuse_ci95_block": boot(reuse),
            "force_rms_mean_meV_A": float(error.mean()), "force_rms_median_meV_A": float(np.median(error)),
            "force_rms_p95_meV_A": float(np.percentile(error, 95)), "force_rms_max_meV_A": float(error.max()),
            "force_rms_ci95_block": boot(error)})
    output.mkdir(parents=True, exist_ok=True)
    (output/"frontier_summary.json").write_text(json.dumps({"input": str(input_csv), "block_length": block,
        "bootstrap_draws": draws, "seed": seed, "groups": summary,
        "interpretation": "Snapshot-level block bootstrap; trajectory is the sampling unit and intervals do not replace independent-trajectory replication."}, indent=2))
    with (output/"frontier_summary.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for tolerance, items in sorted(groups.items()):
            reuse = [1-float(r.get("phi", 0)) for r in items]
            error = [float(r["force_rms_error_meV_A"]) for r in items]
            ax.scatter(reuse, error, s=18, alpha=.7, label=f"{tolerance:g} Å")
        ax.set_xlabel("geometric clean fraction")
        ax.set_ylabel("force RMS error (meV/Å)")
        ax.legend(title="tolerance", fontsize=8)
        fig.tight_layout(); fig.savefig(output/"reuse_error_frontier.png", dpi=180); plt.close(fig)
    except ImportError:
        pass
    return summary


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input", required=True)
    p.add_argument("--output-dir", default="results/active/frontier_analysis")
    p.add_argument("--block-length", type=int, default=10); p.add_argument("--draws", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2027); a=p.parse_args()
    print(json.dumps(analyze(Path(a.input), Path(a.output_dir), a.block_length, a.draws, a.seed), indent=2))
