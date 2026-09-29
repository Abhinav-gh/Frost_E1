"""Frost E1 Stage A: geometric temporal-redundancy characterization only.

This script intentionally performs no dense Jacobian extraction and no exact-vs-
Frost validation. It only updates the per-edge reference cache, records geometric
redundancy statistics, and saves the production trajectory for later Stage B.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from ase.io import Trajectory as ASETrajectory

from frost.e1.geometry import build_edge_geometry_from_ase
from frost.e1.model_interface import load_model
from frost.e1.reference_cache import PerEdgeReferenceCache
from frost.e1.systems import build_system
from frost.e1.trajectory import run_md_trajectory


DEFAULT_TOLERANCES_A = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]


def percentile(values, q):
    return float(np.percentile(values, q)) if values else 0.0


def debug_print(args, message: str):
    if args.debug:
        print(f"[debug] {message}")


def write_plot(rows, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not rows:
        return
    steps = sorted({row["step"] for row in rows})
    tolerances = sorted({row["tolerance_A"] for row in rows})
    fig, axis = plt.subplots(figsize=(8, 4))
    for tolerance in tolerances:
        values = [row["phi"] for row in rows if row["tolerance_A"] == tolerance]
        axis.plot(steps, values, label=f"{tolerance:g} A")
    axis.set(xlabel="production step", ylabel="dirty fraction phi")
    axis.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "phi_vs_timestep.png", dpi=140)
    plt.close(fig)


def run(args):
    if not args.stage_a:
        raise RuntimeError(
            "This script is Stage A only. Use frost.e1.stage_b_accuracy_sampling for Stage B."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tolerances = list(args.tolerances)
    debug_print(args, "Stage A: sampling disabled; Jacobian path disabled")
    debug_print(
        args,
        f"starting Stage-A run: seed={args.seed}, eq={args.equilibration_steps}, prod={args.production_steps}, tolerances={tolerances}",
    )

    atoms = build_system("diamond_si")
    model = load_model("mace-mp-0", device="cuda", dtype="float64")
    caches = {eps: PerEdgeReferenceCache([eps]) for eps in tolerances}
    rows = []
    sampled_steps = []
    sampled_dense_jacobian_seconds = 0.0
    trajectory_path = output_dir / "trajectory.traj"
    trajectory = ASETrajectory(str(trajectory_path), "w")
    production_start = time.perf_counter()

    def callback(current_atoms, step):
        current_atoms = current_atoms.copy()
        trajectory.write(current_atoms)
        edges = build_edge_geometry_from_ase(current_atoms, model.cutoff)
        if step == 0:
            for cache in caches.values():
                cache.initialize(edges, timestep=0)
        stats_by_eps = {eps: caches[eps].step(edges, timestep=step) for eps in tolerances}

        for eps, stats in stats_by_eps.items():
            perturbations = stats["perturbations"]
            edge_delta = [
                value["delta_r"]
                for value in perturbations.values()
                if not value["new"] and np.isfinite(value["delta_r"])
            ]
            rows.append(
                {
                    "step": step,
                    "tolerance_A": eps,
                    "n_edges": stats["n_edges_current"],
                    "n_clean": stats["cache_hits"][eps],
                    "n_dirty": stats["dirty_counts"][eps],
                    "n_new": stats["n_edges_new"],
                    "n_removed": stats["n_edges_removed"],
                    "n_refreshed": stats["refresh_counts"][eps],
                    "phi": stats["dirty_fractions"][eps],
                    "clean_fraction": 1.0 - stats["dirty_fractions"][eps],
                    "mean_delta_r": float(np.mean(edge_delta)) if edge_delta else 0.0,
                    "median_delta_r": percentile(edge_delta, 50),
                    "p90_delta_r": percentile(edge_delta, 90),
                    "p95_delta_r": percentile(edge_delta, 95),
                    "p99_delta_r": percentile(edge_delta, 99),
                    "max_delta_r": float(np.max(edge_delta)) if edge_delta else 0.0,
                    "generation_max": max(stats["generation_by_key"].values(), default=0),
                }
            )

        if args.debug and step in {0, 1, 10, 25, 50, 100, args.production_steps - 1}:
            phi_values = [stats_by_eps[eps]["dirty_fractions"][eps] for eps in tolerances]
            print(
                f"[debug] step={step} phi_range=[{min(phi_values):.4f}, {max(phi_values):.4f}] "
                f"n_edges={stats_by_eps[tolerances[0]]['n_edges_current']}"
            )

    metadata = run_md_trajectory(
        atoms=atoms,
        calculator=model,
        temperature_K=300,
        n_equilibration_steps=args.equilibration_steps,
        n_production_steps=args.production_steps,
        timestep_fs=1.0,
        langevin_gamma=0.01,
        seed=args.seed,
        step_callback=callback,
        log_interval=1,
    )
    trajectory.close()
    production_seconds = time.perf_counter() - production_start

    fieldnames = [
        "step",
        "tolerance_A",
        "n_edges",
        "n_clean",
        "n_dirty",
        "n_new",
        "n_removed",
        "n_refreshed",
        "phi",
        "clean_fraction",
        "mean_delta_r",
        "median_delta_r",
        "p90_delta_r",
        "p95_delta_r",
        "p99_delta_r",
        "max_delta_r",
        "generation_max",
    ]

    with (output_dir / "per_timestep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for eps in tolerances:
        values = [row["phi"] for row in rows if row["tolerance_A"] == eps]
        summaries.append(
            {
                "tolerance_A": eps,
                "mean_phi": float(np.mean(values)),
                "median_phi": float(np.median(values)),
                "p90_phi": percentile(values, 90),
                "p95_phi": percentile(values, 95),
                "max_phi": float(np.max(values)),
                "mean_clean_fraction": float(np.mean([1.0 - value for value in values])),
                "fraction_steps_with_edge_churn": float(
                    np.mean([row["n_new"] + row["n_removed"] > 0 for row in rows if row["tolerance_A"] == eps])
                ),
            }
        )

    summary = {
        "stage": "A",
        "jacobian_enabled": False,
        "sampled_steps": [],
        "sampled_dense_jacobian_seconds": 0.0,
        "metadata": metadata,
        "seed": args.seed,
        "equilibration_steps": args.equilibration_steps,
        "production_steps": args.production_steps,
        "tolerances_A": tolerances,
        "model": "mace-mp-0",
        "system": "diamond_si",
        "production_wall_seconds": production_seconds,
        "summaries": summaries,
        "theta_status": "not measured; clean fraction is a geometric reuse upper bound only",
        "force_status": "Stage A does not compute exact-vs-Frost force errors",
        "trajectory_path": str(trajectory_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_plot(rows, output_dir)
    debug_print(args, f"Stage A complete: wrote {len(rows)} step/tolerance rows to {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frost E1 Stage A: geometry-only temporal redundancy characterization")
    parser.add_argument("--equilibration-steps", type=int, default=100)
    parser.add_argument("--production-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/e1/temporal_redundancy")
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=DEFAULT_TOLERANCES_A,
        help="Geometric tolerances in Å for Stage A; values are displacement thresholds, not meV/Å.",
    )
    parser.add_argument(
        "--stage-a",
        "--no-sampling",
        dest="stage_a",
        action="store_true",
        help="Explicit Stage-A mode: no Jacobian sampling, no dense exact-vs-Frost validation.",
    )
    parser.add_argument("--debug", action="store_true", help="Print concise progress updates")
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=1,
        help="Ignored in Stage A; Stage A never samples Jacobians.",
    )
    parser.set_defaults(stage_a=True)
    run(parser.parse_args())
