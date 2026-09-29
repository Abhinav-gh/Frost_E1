"""Frost E1 Stage B: sparse exact-vs-first-order-Frost validation.

Stage A stores every production configuration, but only aggregate cache
statistics. This program replays the trajectory with the same per-edge cache
rules, records each selected edge's actual reference timestep and geometry, and
then evaluates dense Jacobians only for the unique reference configurations
needed by the requested selected frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from frost.e1.force_error import (
    compute_cached_forces,
    compute_exact_forces,
    extract_layer1_dense_jacobian,
    extract_layer1_local_inputs,
    make_layer1_edge_action,
)
from frost.e1.geometry import (
    build_edge_geometry_from_ase,
    compare_edge_key_sets,
    mace_edge_keys,
)
from frost.e1.model_interface import load_model
from frost.e1.reference_cache import PerEdgeReferenceCache


def debug_print(args, message: str):
    if args.debug:
        print(f"[debug] {message}", flush=True)


def parse_step_list(raw: str) -> list[int]:
    if raw is None or raw == "":
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_tolerance_list(raw: str) -> list[float]:
    if raw is None or raw == "":
        return []
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def infer_trajectory_path(input_csv: Path, directory: Path | None = None) -> Path:
    candidates = [
        input_csv.parent / "trajectory.traj",
        input_csv.parent / "trajectory.xyz",
        input_csv.parent / "production_trajectory.traj",
    ]
    if directory is not None:
        candidates.append(directory / "trajectory.traj")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a Stage-A trajectory file next to the per_timestep.csv. "
        "Provide --trajectory explicitly."
    )


def load_stage_a_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for row in rows:
        row["step"] = int(float(row["step"]))
        row["tolerance_A"] = float(row["tolerance_A"])
        row["phi"] = float(row["phi"])
    return rows


def select_rows(rows: list[dict], steps: list[int], tolerances: list[float]) -> list[dict]:
    selected = []
    if not steps:
        raise ValueError("No steps were provided for Stage B; use --steps 12,40,130")
    if not tolerances:
        raise ValueError("No tolerances were provided for Stage B; use --tolerances 0.001,0.005,0.01")

    step_set = set(steps)
    tol_set = set(tolerances)
    for row in rows:
        if row["step"] in step_set and row["tolerance_A"] in tol_set:
            selected.append(row)
    return selected


def _trajectory_frames(path: Path):
    frames = read(str(path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    return frames


def _replay_reference_states(frames, cutoff: float, tolerances: list[float], selected_steps: list[int]):
    """Replay Stage-A cache transitions and retain only selected-step state."""
    caches = {eps: PerEdgeReferenceCache([eps]) for eps in tolerances}
    selected_states = {eps: {} for eps in tolerances}
    max_step = max(selected_steps)

    for step, atoms in enumerate(frames[: max_step + 1]):
        edges = build_edge_geometry_from_ase(atoms, cutoff)
        if step == 0:
            for cache in caches.values():
                cache.initialize(edges, timestep=0)
        stats_by_eps = {eps: caches[eps].step(edges, timestep=step) for eps in tolerances}
        if step not in selected_steps:
            continue

        for eps, cache in caches.items():
            selected_states[eps][step] = {
                "stats": stats_by_eps[eps],
                "references": {
                    key: {
                        "ref_timestep": entry.ref_timestep,
                        "ref_r_ij": entry.ref_r_ij.copy(),
                    }
                    for key, entry in cache._entries.items()
                },
            }
    return selected_states


def _keyed_dense_reference(model, atoms, expected_keys, dense_cache, timestep):
    """Extract one dense snapshot and return its values indexed by edge key."""
    cached = dense_cache.get(timestep)
    if cached is not None:
        return cached

    dense = extract_layer1_dense_jacobian(model, atoms, chunk_size=16)
    unit_shifts = extract_layer1_local_inputs(model, atoms)["unit_shifts"]
    keys = mace_edge_keys(dense["edge_index"], unit_shifts)
    comparison = compare_edge_key_sets(expected_keys, keys)
    if not comparison["matched"]:
        raise RuntimeError(
            f"Stage-B edge mapping mismatch at reference step {timestep}: "
            f"ASE={comparison['ase_count']} MACE={comparison['mace_count']} "
            f"missing_from_mace={comparison['missing_from_mace'][:5]} "
            f"missing_from_ase={comparison['missing_from_ase'][:5]}"
        )
    keyed = {
        key: {
            "g": dense["g_ref"][index].detach().clone(),
            "J": dense["J_ref"][index].detach().clone(),
            "vectors": dense["vectors_ref"][index].detach().clone(),
        }
        for index, key in enumerate(keys)
    }
    dense_cache[timestep] = keyed
    return keyed


def _keyed_action_reference(model, atoms, expected_keys, action_cache, timestep):
    """Capture reference messages and bind JVP/VJP actions without dense J."""
    cached = action_cache.get(timestep)
    if cached is not None:
        return cached
    local = extract_layer1_local_inputs(model, atoms)
    keys = mace_edge_keys(local["edge_index"], local["unit_shifts"])
    comparison = compare_edge_key_sets(expected_keys, keys)
    if not comparison["matched"]:
        raise RuntimeError(
            f"Stage-B edge mapping mismatch at reference step {timestep}: {comparison}"
        )
    keyed = {
        key: {
            "g": local["mji"][index].detach().clone(),
            "vectors": local["vectors"][index].detach().clone(),
            "action": make_layer1_edge_action(model, local, index),
        }
        for index, key in enumerate(keys)
    }
    action_cache[timestep] = keyed
    return keyed


def _reference_plan(selected_states, selected_steps, tolerances):
    pairs = set()
    per_frame = {}
    for tolerance in tolerances:
        for step in selected_steps:
            state = selected_states[tolerance][step]
            dirty = state["stats"]["dirty_keys_by_eps"][tolerance]
            for key, entry in state["references"].items():
                if key in dirty:
                    continue
                pair = (entry["ref_timestep"], key)
                pairs.add(pair)
                per_frame.setdefault(entry["ref_timestep"], set()).add(key)
    counts = [len(keys) for keys in per_frame.values()]
    return {
        "selected_rows": len(selected_steps) * len(tolerances),
        "unique_reference_frames": len(per_frame),
        "unique_edge_reference_pairs": len(pairs),
        "max_requested_edges_per_reference_frame": max(counts, default=0),
        "mean_requested_edges_per_reference_frame": float(np.mean(counts)) if counts else 0.0,
        "requested_edges_by_reference_frame": {
            str(step): len(keys) for step, keys in sorted(per_frame.items())
        },
    }


def run(args):
    run_started = time.perf_counter()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_stage_a_rows(input_path)
    if not rows:
        raise ValueError(f"No rows found in Stage-A CSV: {input_path}")
    debug_print(args, f"loaded {len(rows)} Stage-A rows from {input_path}")

    trajectory_path = Path(args.trajectory) if args.trajectory else infer_trajectory_path(input_path)
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Stage-A trajectory not found: {trajectory_path}")

    selected_steps = parse_step_list(args.steps)
    selected_tolerances = parse_tolerance_list(args.tolerances)
    if args.pilot:
        selected_steps = [40]
        selected_tolerances = [0.01, 0.02, 0.05]
        print("[stage-b] pilot mode: step=40 tolerances=0.01,0.02,0.05 A", flush=True)
    selected = select_rows(rows, selected_steps, selected_tolerances)

    if not selected:
        raise ValueError(
            "Stage-B selection matched no rows. Check that the selected steps and tolerances exist in the Stage-A CSV."
        )

    # Validate trajectory coverage for requested step indices.
    traj = _trajectory_frames(trajectory_path)
    debug_print(args, f"loaded {len(traj)} trajectory frames from {trajectory_path}")
    available_steps = list(range(len(traj)))
    missing_steps = sorted(set(selected_steps) - set(available_steps))
    if missing_steps:
        raise ValueError(
            f"Requested Stage-B step indices do not exist in the Stage-A trajectory: {missing_steps}"
        )

    print(f"[stage-b] replaying {max(selected_steps) + 1} trajectory frames", flush=True)
    model = load_model("mace-mp-0", device=args.device, dtype="float64")
    debug_print(args, f"loaded MACE model on {args.device}; selected_steps={selected_steps}")
    selected_states = _replay_reference_states(
        traj, model.cutoff, selected_tolerances, selected_steps
    )
    replay_seconds = time.perf_counter() - run_started
    debug_print(args, "cache replay complete; beginning sparse exact/Frost evaluations")
    if args.plan_only:
        plan = _reference_plan(selected_states, selected_steps, selected_tolerances)
        plan.update({
            "stage": "B",
            "plan_only": True,
            "input_csv": str(input_path),
            "trajectory_path": str(trajectory_path),
            "selected_steps": selected_steps,
            "selected_tolerances_A": selected_tolerances,
            "cache_replay_seconds": replay_seconds,
        })
        (output_dir / "plan.json").write_text(json.dumps(plan, indent=2))
        print(json.dumps(plan, indent=2))
        return
    action_cache = {}
    result_rows = []
    exact_cache = {}
    action_reference_frames = 0
    action_started = time.perf_counter()

    for step in selected_steps:
        atoms = traj[step]
        debug_print(args, f"step {step}: extracting current edge inputs and exact forces")
        current_edges = build_edge_geometry_from_ase(atoms, model.cutoff)
        current_local = extract_layer1_local_inputs(model, atoms)
        current_vectors = current_local["vectors"].detach()
        current_keys = mace_edge_keys(current_local["edge_index"], current_local["unit_shifts"])
        current_comparison = compare_edge_key_sets(current_edges.keys(), current_keys)
        if not current_comparison["matched"]:
            raise RuntimeError(
                f"Stage-B edge mapping mismatch at selected step {step}: {current_comparison}"
            )
        current_by_key = {key: index for index, key in enumerate(current_keys)}
        if step not in exact_cache:
            exact_cache[step] = compute_exact_forces(atoms, model)
        exact_energy, exact_forces = exact_cache[step]

        for tolerance in selected_tolerances:
            evaluation_started = time.perf_counter()
            state = selected_states[tolerance][step]
            references = state["references"]
            reference_timesteps = {entry["ref_timestep"] for entry in references.values()}
            debug_print(
                args,
                f"step {step}, tolerance={tolerance:g} A: "
                f"clean={state['stats']['cache_hits'][tolerance]} "
                f"dirty={state['stats']['dirty_counts'][tolerance]} "
                f"reference_frames={len(reference_timesteps)}",
            )
            reference_by_timestep = {}
            for reference_step in reference_timesteps:
                reference_atoms = traj[reference_step]
                reference_edges = build_edge_geometry_from_ase(reference_atoms, model.cutoff)
                reference_by_timestep[reference_step] = _keyed_action_reference(
                    model,
                    reference_atoms,
                    reference_edges.keys(),
                    action_cache,
                    reference_step,
                )
            action_reference_frames = len(action_cache)

            clean_mask = torch.zeros(len(current_keys), dtype=torch.bool)
            g_reference = current_local["mji"].detach().clone()
            vectors_reference = current_vectors.clone()
            action_fns = [None] * len(current_keys)
            for key, entry in references.items():
                index = current_by_key[key]
                reference_values = reference_by_timestep[entry["ref_timestep"]][key]
                displacement = np.linalg.norm(
                    current_edges[key]["r_ij"] - entry["ref_r_ij"]
                )
                if key not in state["stats"]["dirty_keys_by_eps"][tolerance] and displacement <= tolerance:
                    clean_mask[index] = True
                    g_reference[index] = reference_values["g"]
                    vectors_reference[index] = reference_values["vectors"]
                    action_fns[index] = reference_values["action"]

            frost_energy, frost_forces = compute_cached_forces(
                atoms,
                model,
                clean_mask,
                g_reference,
                mode=3,
                vectors=current_vectors,
                vectors_ref=vectors_reference,
                action_fns=action_fns,
            )
            force_delta = np.asarray(frost_forces) - np.asarray(exact_forces)
            force_norms = np.linalg.norm(force_delta, axis=1) * 1000.0
            result_rows.append(
                {
                    "step": step,
                    "tolerance_A": tolerance,
                    "phi": state["stats"]["dirty_fractions"][tolerance],
                    "n_edges": state["stats"]["n_edges_current"],
                    "n_clean": int(clean_mask.sum().item()),
                    "n_dirty": int((~clean_mask).sum().item()),
                    "exact_energy_eV": float(exact_energy),
                    "frost_energy_eV": float(frost_energy),
                    "energy_error_eV": abs(float(frost_energy - exact_energy)),
                    "force_max_error_meV_A": float(force_norms.max()),
                    "force_mean_error_meV_A": float(force_norms.mean()),
                    "force_rms_error_meV_A": float(np.sqrt(np.mean(force_delta ** 2)) * 1000.0),
                    "relative_force_error": float(np.linalg.norm(force_delta) / max(np.linalg.norm(exact_forces), 1e-30)),
                    "finite": bool(
                        np.isfinite(exact_energy)
                        and np.isfinite(frost_energy)
                        and np.isfinite(exact_forces).all()
                        and np.isfinite(frost_forces).all()
                    ),
                    "status": "ok",
                    "n_reference_configurations_used": len(reference_timesteps),
                    "action_evaluation_seconds": time.perf_counter() - evaluation_started,
                }
            )
        print(f"[stage-b] evaluated step {step}; action_reference_snapshots={action_reference_frames}", flush=True)

    action_seconds = time.perf_counter() - action_started

    result_fields = list(result_rows[0].keys())
    with (output_dir / "accuracy_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(result_rows)

    grouped = []
    for tolerance in selected_tolerances:
        tolerance_rows = [row for row in result_rows if row["tolerance_A"] == tolerance]
        def distribution(field):
            values = np.asarray([row[field] for row in tolerance_rows], dtype=float)
            return {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        grouped.append(
            {
                "tolerance_A": tolerance,
                "n_rows": len(tolerance_rows),
                "phi": distribution("phi"),
                "clean_fraction": distribution("clean_fraction") if "clean_fraction" in tolerance_rows[0] else {
                    "mean": float(np.mean([row["n_clean"] / row["n_edges"] for row in tolerance_rows])),
                    "median": float(np.median([row["n_clean"] / row["n_edges"] for row in tolerance_rows])),
                    "std": float(np.std([row["n_clean"] / row["n_edges"] for row in tolerance_rows])),
                    "min": float(np.min([row["n_clean"] / row["n_edges"] for row in tolerance_rows])),
                    "max": float(np.max([row["n_clean"] / row["n_edges"] for row in tolerance_rows])),
                },
                "energy_error_eV": distribution("energy_error_eV"),
                "force_rms_error_meV_A": distribution("force_rms_error_meV_A"),
                "force_max_error_meV_A": distribution("force_max_error_meV_A"),
                "relative_force_error": distribution("relative_force_error"),
                "action_evaluation_seconds": distribution("action_evaluation_seconds"),
                "n_clean": distribution("n_clean"),
                "n_dirty": distribution("n_dirty"),
                "n_reference_configurations_used": distribution("n_reference_configurations_used"),
                "all_finite": bool(all(row["finite"] for row in tolerance_rows)),
            }
        )

    summary = {
        "stage": "B",
        "input_csv": str(input_path),
        "trajectory_path": str(trajectory_path),
        "selected_steps": selected_steps,
        "selected_tolerances_A": selected_tolerances,
        "n_selected_rows": len(result_rows),
        "reference_configurations": sorted(action_cache),
        "reference_configuration_count": len(action_cache),
        "dense_jacobian_enabled": False,
        "action_evaluation_seconds": action_seconds,
        "total_stage_b_runtime_seconds": time.perf_counter() - run_started,
        "pilot_mode": bool(args.pilot),
        "accuracy_results_csv": str(output_dir / "accuracy_results.csv"),
        "summaries_by_tolerance": grouped,
        "force_error_executed": True,
        "all_finite": bool(all(row["finite"] for row in result_rows)),
        "selected_rows": result_rows,
        "notes": (
            "Stage B is intentionally sparse: dense J_ref/exact MACE/first-order Frost is applied only "
            "for the selected Stage-A snapshots and their required reference states."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[stage-b] complete: rows={len(result_rows)} "
        f"reference_frames={len(action_cache)} "
        f"action_seconds={action_seconds:.1f} "
        f"total_seconds={summary['total_stage_b_runtime_seconds']:.1f}",
        flush=True,
    )
    debug_print(args, f"wrote {len(result_rows)} accuracy rows and grouped summary to {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frost E1 Stage B: sparse snapshot validation")
    parser.add_argument("--input", required=True, help="Path to Stage-A per_timestep.csv")
    parser.add_argument("--trajectory", default=None, help="Optional path to the saved Stage-A trajectory.traj")
    parser.add_argument("--steps", default="40", help="Comma-separated production steps, e.g. 40,130")
    parser.add_argument(
        "--tolerances",
        default="0.01,0.02,0.05",
        help="Comma-separated geometric tolerances in Å, e.g. 0.001,0.005,0.01,0.02,0.05",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/e1/stageB_snapshot_accuracy")
    parser.add_argument("--debug", action="store_true", help="Print concise Stage-B progress messages")
    parser.add_argument("--plan-only", action="store_true", help="Replay cache geometry and report sparse derivative dependencies only")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run the bounded accuracy pilot: step 40 with tolerances 0.01, 0.02, and 0.05 A.",
    )
    run(parser.parse_args())
