"""Bounded E1 temporal-redundancy characterization on diamond Si.

This runner measures geometric cache redundancy every MD production step. It
also performs sampled full first-order Frost comparisons. The sampled Frost
cache refreshes dense g/J snapshots at the sampling cadence; it is deliberately
not the final production cache manager.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from frost.e1.force_error import (
    compute_cached_forces,
    compute_exact_forces,
    extract_layer1_dense_jacobian,
    extract_layer1_local_inputs,
)
from frost.e1.geometry import build_edge_geometry_from_ase
from frost.e1.geometry import canonical_edge_key, mace_edge_keys
from frost.e1.model_interface import load_model
from frost.e1.reference_cache import PerEdgeReferenceCache
from frost.e1.systems import build_system
from frost.e1.trajectory import run_md_trajectory


def percentile(values, q):
    return float(np.percentile(values, q)) if values else 0.0


def write_plot(rows, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps = sorted({row["step"] for row in rows})
    tolerances = sorted({row["tolerance_A"] for row in rows})
    fig, axis = plt.subplots(figsize=(8, 4))
    for tolerance in tolerances:
        values = [
            row["phi"] for row in rows
            if row["tolerance_A"] == tolerance
        ]
        axis.plot(steps, values, label=f"{tolerance:g} A")
    axis.set(xlabel="production step", ylabel="dirty fraction phi")
    axis.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "phi_vs_timestep.png", dpi=140)
    plt.close(fig)


def debug_print(args, message: str):
    if args.debug:
        print(f"[debug] {message}")


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tolerances = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    debug_print(args, f"starting run: seed={args.seed}, eq={args.equilibration_steps}, prod={args.production_steps}, sample_interval={args.sample_interval}")
    atoms = build_system("diamond_si")
    debug_print(args, "building diamond Si system")
    model = load_model("mace-mp-0", device="cuda", dtype="float64")
    debug_print(args, f"loaded model; cutoff={model.cutoff:.3f} A; tolerances={tolerances}")
    caches = {eps: PerEdgeReferenceCache([eps]) for eps in tolerances}
    rows = []
    force_rows = []
    force_mismatch_rows = []
    previous_edges = None
    previous_edge_geometry = None
    previous_positions = None
    sampled_reference = None
    production_start = time.perf_counter()
    dense_seconds = 0.0
    sampled_steps = []

    def callback(current_atoms, step):
        nonlocal previous_edges, previous_edge_geometry, previous_positions, sampled_reference, dense_seconds
        edges = build_edge_geometry_from_ase(current_atoms, model.cutoff)
        if previous_edge_geometry is None:
            step_edge_delta_values = []
        else:
            step_edge_delta_values = [
                float(np.linalg.norm(geom["r_ij"] - previous_edge_geometry[key]["r_ij"]))
                for key, geom in edges.items()
                if key in previous_edge_geometry
            ]
        if previous_positions is None:
            step_delta_values = [0.0]
        else:
            step_delta_values = [
                float(np.linalg.norm(current_atoms.positions[j] - previous_positions[j]))
                for j in range(len(current_atoms))
            ]
        previous_positions = current_atoms.get_positions().copy()

        for cache in caches.values():
            if step == 0:
                cache.initialize(edges, timestep=0)
        stats_by_eps = {
            eps: caches[eps].step(edges, timestep=step)
            for eps in tolerances
        }
        previous_edges = set(edges)
        previous_edge_geometry = edges

        for eps, stats in stats_by_eps.items():
            perturbations = stats["perturbations"]
            edge_delta = [
                value["delta_r"] for value in perturbations.values()
                if not value["new"] and np.isfinite(value["delta_r"])
            ]
            rows.append({
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
                "mean_step_delta_r": float(np.mean(step_edge_delta_values)) if step_edge_delta_values else 0.0,
                "median_step_delta_r": percentile(step_edge_delta_values, 50),
                "p90_step_delta_r": percentile(step_edge_delta_values, 90),
                "p95_step_delta_r": percentile(step_edge_delta_values, 95),
                "p99_step_delta_r": percentile(step_edge_delta_values, 99),
                "max_step_delta_r": float(np.max(step_edge_delta_values)) if step_edge_delta_values else 0.0,
                "mean_atom_step_displacement": float(np.mean(step_delta_values)),
                "max_atom_step_displacement": float(np.max(step_delta_values)),
                "generation_max": max(stats["generation_by_key"].values(), default=0),
            })

        if step % args.sample_interval != 0:
            if args.debug and step in {0, 1, 5, 10, 25, 50, 100}:
                phi_values = [
                    stats["dirty_fractions"][eps]
                    for eps, stats in stats_by_eps.items()
                ]
                print(
                    f"[debug] step={step} phi_range=[{min(phi_values):.4f}, {max(phi_values):.4f}] "
                    f"n_edges={stats_by_eps[min(tolerances)]['n_edges_current']}"
                )
            return

        sample_start = time.perf_counter()
        debug_print(args, f"sampling exact vs Frost at step={step}")
        local = extract_layer1_local_inputs(model, current_atoms)
        dense = extract_layer1_dense_jacobian(model, current_atoms, chunk_size=16)
        dense_seconds += time.perf_counter() - sample_start
        current_g = local["mji"].detach()
        current_vectors = local["vectors"].detach()
        mace_keys = mace_edge_keys(
            local["edge_index"], local["unit_shifts"]
        )
        if sampled_reference is None:
            sampled_reference = {
                eps: {
                    key: {
                        "g": current_g[index].clone(),
                        "J": dense["J_ref"][index].clone(),
                        "vectors": current_vectors[index].clone(),
                    }
                    for index, key in enumerate(mace_keys)
                }
                for eps in tolerances
            }
            return

        exact_energy, exact_forces = compute_exact_forces(current_atoms, model)
        for eps in tolerances:
            reference_by_key = sampled_reference[eps]
            clean_mask = torch.zeros(current_g.shape[0], dtype=torch.bool)
            g_reference = current_g.clone()
            J_reference = dense["J_ref"].clone()
            vectors_reference = current_vectors.clone()
            for index, key in enumerate(mace_keys):
                entry = reference_by_key.get(key)
                if entry is None:
                    continue
                g_reference[index] = entry["g"]
                J_reference[index] = entry["J"]
                vectors_reference[index] = entry["vectors"]
                if float(torch.linalg.vector_norm(current_vectors[index] - entry["vectors"])) <= eps:
                    clean_mask[index] = True
            frost_energy, frost_forces = compute_cached_forces(
                current_atoms,
                model,
                clean_mask,
                g_reference,
                mode=2,
                J_ref=J_reference,
                vectors=current_vectors,
                vectors_ref=vectors_reference,
            )
            delta = frost_forces - exact_forces
            force_rows.append({
                "step": step,
                "tolerance_A": eps,
                "exact_energy": float(exact_energy),
                "frost_energy": float(frost_energy),
                "energy_error": abs(float(frost_energy - exact_energy)),
                "force_max_error": float(np.abs(delta).max()),
                "force_mean_error": float(np.abs(delta).mean()),
                "force_rms_error": float(np.sqrt(np.mean(delta ** 2))),
                "relative_force_error": float(np.linalg.norm(delta) / max(np.linalg.norm(exact_forces), 1e-30)),
                "finite": bool(np.isfinite(frost_energy) and np.isfinite(frost_forces).all()),
            })
            for index, key in enumerate(mace_keys):
                if not clean_mask[index]:
                    reference_by_key[key] = {
                        "g": current_g[index].clone(),
                        "J": dense["J_ref"][index].clone(),
                        "vectors": current_vectors[index].clone(),
                    }
        sampled_steps.append(step)

    debug_print(args, "starting MD trajectory")
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
    production_seconds = time.perf_counter() - production_start
    debug_print(args, f"MD complete; runtime={production_seconds:.1f}s; sampled_steps={len(sampled_steps)}")

    with (output_dir / "per_timestep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "force_samples.csv").open("w", newline="") as handle:
        if force_rows:
            writer = csv.DictWriter(handle, fieldnames=force_rows[0].keys())
            writer.writeheader()
            writer.writerows(force_rows)
    with (output_dir / "force_mismatches.csv").open("w", newline="") as handle:
        if force_mismatch_rows:
            writer = csv.DictWriter(handle, fieldnames=force_mismatch_rows[0].keys())
            writer.writeheader()
            writer.writerows(force_mismatch_rows)

    summaries = []
    for eps in tolerances:
        values = [row["phi"] for row in rows if row["tolerance_A"] == eps]
        force_values = [row["force_rms_error"] for row in force_rows if row["tolerance_A"] == eps]
        summaries.append({
            "tolerance_A": eps,
            "mean_phi": float(np.mean(values)),
            "median_phi": float(np.median(values)),
            "p90_phi": percentile(values, 90),
            "p95_phi": percentile(values, 95),
            "max_phi": float(np.max(values)),
            "mean_clean_fraction": float(np.mean([1.0 - value for value in values])),
            "fraction_steps_with_edge_churn": float(np.mean([
                row["n_new"] + row["n_removed"] > 0
                for row in rows if row["tolerance_A"] == eps
            ])),
            "mean_sampled_force_rms_error": float(np.mean(force_values)) if force_values else None,
            "n_force_samples": len(force_values),
        })
    summary = {
        "metadata": metadata,
        "seed": args.seed,
        "equilibration_steps": args.equilibration_steps,
        "production_steps": args.production_steps,
        "sample_interval": args.sample_interval,
        "sampled_steps": sampled_steps,
        "force_comparison_mismatches": force_mismatch_rows,
        "production_wall_seconds": production_seconds,
        "sampled_dense_jacobian_seconds": dense_seconds,
        "tolerances_A": tolerances,
        "summaries": summaries,
        "theta_status": "not measured; clean fraction is only a geometric reuse upper bound",
        "force_status": "sampled dense snapshots refreshed at sample cadence; not a production per-edge refresh engine",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_plot(rows, output_dir)
    if args.debug:
        print("[debug] finished writing summary and plots")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--equilibration-steps", type=int, default=100)
    parser.add_argument("--production-steps", type=int, default=500)
    parser.add_argument("--sample-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/e1/temporal_redundancy")
    parser.add_argument("--debug", action="store_true", help="print concise progress updates during the run")
    run(parser.parse_args())
