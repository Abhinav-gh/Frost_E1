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


def stratified_steps(rows: list[dict], tolerances: list[float], count: int) -> list[int]:
    """Choose approximately even time samples, including trajectory endpoints."""
    candidates = sorted({row["step"] for row in rows if row["tolerance_A"] in set(tolerances)})
    if count < 1:
        raise ValueError("sample_count must be positive")
    if not candidates:
        raise ValueError("No Stage-A steps match the requested tolerances")
    count = min(count, len(candidates))
    indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return [candidates[index] for index in sorted(set(indices.tolist()))]


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


def _keyed_dense_reference(model, atoms, expected_keys, dense_cache, timestep,
                           chunk_size=64, requested_keys=None):
    """Extract one dense snapshot and return its values indexed by edge key."""
    cached = dense_cache.setdefault(timestep, {})
    requested = set(expected_keys if requested_keys is None else requested_keys)
    missing = requested - set(cached)
    if not missing:
        return cached

    local = extract_layer1_local_inputs(model, atoms)
    all_keys = mace_edge_keys(local["edge_index"], local["unit_shifts"])
    comparison = compare_edge_key_sets(expected_keys, all_keys)
    if not comparison["matched"]:
        raise RuntimeError(f"Stage-B edge mapping mismatch at dense reference step {timestep}: {comparison}")
    missing_indices = [index for index, key in enumerate(all_keys) if key in missing]
    missing_keys = [all_keys[index] for index in missing_indices]
    if set(missing_keys) != missing:
        raise RuntimeError(
            f"Stage-B requested edge mapping mismatch at reference step {timestep}: "
            f"requested={len(missing)} matched={len(missing_keys)}"
        )
    dense = extract_layer1_dense_jacobian(
        model, atoms, chunk_size=chunk_size, edge_indices=missing_indices
    )
    keys = mace_edge_keys(
        dense["edge_index"], local["unit_shifts"][dense["selected_edge_indices"]]
    )
    keyed = {
        key: {
            "g": dense["g_ref"][index].detach().clone(),
            "J": dense["J_ref"][index].detach().clone(),
            "vectors": dense["vectors_ref"][index].detach().clone(),
        }
        for index, key in enumerate(keys)
    }
    cached.update(keyed)
    return cached


def _keyed_action_reference(model, atoms, expected_keys, action_cache, timestep,
                            with_actions=True, requested_keys=None):
    """Capture reference messages and bind JVP/VJP actions without dense J."""
    cached = action_cache.setdefault(timestep, {})
    requested = set(expected_keys if requested_keys is None else requested_keys)
    missing = requested - set(cached)
    if not missing:
        return cached
    local = extract_layer1_local_inputs(model, atoms)
    keys = mace_edge_keys(local["edge_index"], local["unit_shifts"])
    comparison = compare_edge_key_sets(expected_keys, keys)
    if not comparison["matched"]:
        raise RuntimeError(
            f"Stage-B edge mapping mismatch at reference step {timestep}: {comparison}"
        )
    indices = {key: index for index, key in enumerate(keys)}
    if not missing.issubset(indices):
        raise RuntimeError(f"Stage-B requested edge missing from reference timestep {timestep}")
    keyed = {
        key: {
            "g": local["mji"][indices[key]].detach().clone(),
            "vectors": local["vectors"][indices[key]].detach().clone(),
            "action": make_layer1_edge_action(model, local, indices[key]) if with_actions else None,
        }
        for key in missing
    }
    cached.update(keyed)
    return cached


def _keyed_batched_reference(model, atoms, expected_keys, batched_cache,
                             node_attrs_cache, timestep, requested_keys=None):
    """Retain only requested reference edge inputs for batched JVP/VJP mode."""
    cached = batched_cache.setdefault(timestep, {})
    requested = set(expected_keys if requested_keys is None else requested_keys)
    missing = requested - set(cached)
    if not missing:
        return cached
    local = extract_layer1_local_inputs(model, atoms)
    keys = mace_edge_keys(local["edge_index"], local["unit_shifts"])
    comparison = compare_edge_key_sets(expected_keys, keys)
    if not comparison["matched"]:
        raise RuntimeError(f"Stage-B edge mapping mismatch at batched reference step {timestep}: {comparison}")
    indices = {key: index for index, key in enumerate(keys)}
    if not missing.issubset(indices):
        raise RuntimeError(f"Stage-B requested edge missing from batched reference timestep {timestep}")
    missing_list = list(missing)
    selected = torch.as_tensor(
        [indices[key] for key in missing_list], dtype=torch.long,
        device=local["edge_index"].device,
    )
    senders = local["edge_index"][0, selected]
    sources = local["node_feats"][senders]
    edge_indices = local["edge_index"][:, selected].T
    keyed = {
        key: {
            "g": local["mji"][selected[position]].detach().clone(),
            "vectors": local["vectors"][selected[position]].detach().clone(),
            "source_features": sources[position].detach().clone(),
            "edge_indices": edge_indices[position].detach().clone(),
        }
        for position, key in enumerate(missing_list)
    }
    node_attrs_cache[timestep] = local["node_attrs"].detach()
    cached.update(keyed)
    return cached


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
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
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
    elif getattr(args, "sample_count", 0):
        if selected_steps:
            raise ValueError("Use either --steps or --sample-count, not both")
        selected_steps = stratified_steps(rows, selected_tolerances, args.sample_count)
        print(f"[stage-b] stratified sampling selected {len(selected_steps)} steps", flush=True)
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
    model = load_model(
        getattr(args, "model", "mace-mp-0"),
        device=args.device,
        dtype=getattr(args, "dtype", "float64"),
    )
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
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
    dense_cache = {}
    batched_cache = {}
    batched_node_attrs = {}
    cache_mode = getattr(args, "cache_mode", 3)
    reference_cache = {2: dense_cache, 3: action_cache, 4: batched_cache}.get(cache_mode, action_cache)
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
            dirty_keys = state["stats"]["dirty_keys_by_eps"][tolerance]
            clean_reference_keys = {
                key for key, entry in references.items()
                if key not in dirty_keys
                and np.linalg.norm(current_edges[key]["r_ij"] - entry["ref_r_ij"]) <= tolerance
            }
            reference_timesteps = {
                references[key]["ref_timestep"] for key in clean_reference_keys
            }
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
                requested_keys = {
                    key for key in clean_reference_keys
                    if references[key]["ref_timestep"] == reference_step
                }
                if cache_mode == 2:
                    reference_by_timestep[reference_step] = _keyed_dense_reference(
                        model, reference_atoms, reference_edges.keys(), dense_cache,
                        reference_step, chunk_size=getattr(args, "jacobian_chunk_size", 64),
                        requested_keys=requested_keys,
                    )
                elif cache_mode == 4:
                    reference_by_timestep[reference_step] = _keyed_batched_reference(
                        model, reference_atoms, reference_edges.keys(), batched_cache,
                        batched_node_attrs, reference_step, requested_keys=requested_keys,
                    )
                else:
                    reference_by_timestep[reference_step] = _keyed_action_reference(
                        model, reference_atoms, reference_edges.keys(), action_cache,
                        reference_step, with_actions=(cache_mode == 3),
                        requested_keys=requested_keys,
                    )
            action_reference_frames = len(reference_cache)

            clean_mask = torch.zeros(len(current_keys), dtype=torch.bool)
            g_reference = current_local["mji"].detach().clone()
            vectors_reference = current_vectors.clone()
            action_fns = [None] * len(current_keys)
            J_reference = (torch.zeros(
                (len(current_keys), current_local["mji"].shape[1], 3),
                dtype=current_vectors.dtype, device=current_vectors.device,
            ) if cache_mode == 2 else None)
            source_features_reference = (torch.zeros(
                (len(current_keys), current_local["node_feats"].shape[1]),
                dtype=current_local["node_feats"].dtype, device=current_vectors.device,
            ) if cache_mode == 4 else None)
            edge_indices_reference = (torch.zeros(
                (len(current_keys), 2), dtype=torch.long, device=current_vectors.device,
            ) if cache_mode == 4 else None)
            reference_ages = []
            for key, entry in references.items():
                index = current_by_key[key]
                displacement = np.linalg.norm(
                    current_edges[key]["r_ij"] - entry["ref_r_ij"]
                )
                if key not in dirty_keys and displacement <= tolerance:
                    reference_values = reference_by_timestep[entry["ref_timestep"]][key]
                    clean_mask[index] = True
                    reference_ages.append(step - entry["ref_timestep"])
                    g_reference[index] = reference_values["g"]
                    vectors_reference[index] = reference_values["vectors"]
                    if cache_mode == 2:
                        J_reference[index] = reference_values["J"]
                    elif cache_mode == 4:
                        source_features_reference[index] = reference_values["source_features"]
                        edge_indices_reference[index] = reference_values["edge_indices"]
                    else:
                        action_fns[index] = reference_values["action"]

            message_audit = {}
            batched_action = None
            if cache_mode == 4:
                if reference_timesteps:
                    node_attrs_reference = batched_node_attrs[next(iter(reference_timesteps))]
                else:
                    node_attrs_reference = current_local["node_attrs"].detach()
                batched_action = {
                    "model": model.calculator.models[0],
                    "source_features": source_features_reference,
                    "edge_indices": edge_indices_reference,
                    "node_attrs": node_attrs_reference,
                    "chunk_size": getattr(args, "batched_action_chunk_size", 256),
                }
            frost_energy, frost_forces = compute_cached_forces(
                atoms,
                model,
                clean_mask,
                g_reference,
                mode=cache_mode,
                J_ref=J_reference,
                vectors=current_vectors,
                vectors_ref=vectors_reference,
                action_fns=action_fns,
                batched_action=batched_action,
                audit_capture=message_audit,
            )
            force_delta = np.asarray(frost_forces) - np.asarray(exact_forces)
            force_norms = np.linalg.norm(force_delta, axis=1) * 1000.0
            message_delta = np.linalg.norm(message_audit["mji_used"] - message_audit["mji_exact"], axis=1)
            clean_cpu = message_audit["clean_mask"]
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
                    "force_atom_p95_error_meV_A": float(np.percentile(force_norms, 95)),
                    "force_atom_p99_error_meV_A": float(np.percentile(force_norms, 99)),
                    "edge_message_clean_rms": float(np.sqrt(np.mean(message_delta[clean_cpu] ** 2))) if clean_cpu.any() else 0.0,
                    "edge_message_clean_max": float(message_delta[clean_cpu].max()) if clean_cpu.any() else 0.0,
                    "edge_message_dirty_max": float(message_delta[~clean_cpu].max()) if (~clean_cpu).any() else 0.0,
                    "reference_age_mean_steps": float(np.mean(reference_ages)) if reference_ages else 0.0,
                    "reference_age_max_steps": int(max(reference_ages, default=0)),
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
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
        peak_cuda_allocated_bytes = int(torch.cuda.max_memory_allocated())
    else:
        peak_cuda_allocated_bytes = None

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
                "force_atom_p95_error_meV_A": distribution("force_atom_p95_error_meV_A"),
                "force_atom_p99_error_meV_A": distribution("force_atom_p99_error_meV_A"),
                "edge_message_clean_rms": distribution("edge_message_clean_rms"),
                "edge_message_clean_max": distribution("edge_message_clean_max"),
                "edge_message_dirty_max": distribution("edge_message_dirty_max"),
                "reference_age_mean_steps": distribution("reference_age_mean_steps"),
                "reference_age_max_steps": distribution("reference_age_max_steps"),
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
        "reference_configurations": sorted(reference_cache),
        "reference_configuration_count": len(reference_cache),
        "dense_jacobian_enabled": cache_mode == 2,
        "jacobian_chunk_size": getattr(args, "jacobian_chunk_size", 64) if cache_mode == 2 else None,
        "batched_action_chunk_size": getattr(args, "batched_action_chunk_size", 256) if cache_mode == 4 else None,
        "peak_cuda_allocated_bytes": peak_cuda_allocated_bytes,
        "cache_mode": cache_mode,
        "action_evaluation_seconds": action_seconds,
        "total_stage_b_runtime_seconds": time.perf_counter() - run_started,
        "pilot_mode": bool(args.pilot),
        "sampling": "stratified" if getattr(args, "sample_count", 0) else "explicit_steps",
        "sample_count_requested": getattr(args, "sample_count", 0),
        "model": getattr(args, "model", "mace-mp-0"),
        "device": model.device,
        "dtype": getattr(args, "dtype", "float64"),
        "accuracy_results_csv": str(output_dir / "accuracy_results.csv"),
        "summaries_by_tolerance": grouped,
        "force_error_executed": True,
        "all_finite": bool(all(row["finite"] for row in result_rows)),
        "selected_rows": result_rows,
        "notes": (
            "Stage B evaluates exact MACE and the selected reuse mode only for sampled snapshots and required "
            "references. Modes 2 and 4 are batched first-order alternatives (dense Jacobian and JVP/VJP); "
            "mode 3 is serial per-edge JVP/VJP."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[stage-b] complete: rows={len(result_rows)} "
        f"reference_frames={len(reference_cache)} "
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
    parser.add_argument("--steps", default="", help="Comma-separated production steps, e.g. 40,130")
    parser.add_argument("--sample-count", type=int, default=0, help="Evenly sample this many trajectory steps; mutually exclusive with --steps")
    parser.add_argument(
        "--tolerances",
        default="0.01,0.02,0.05",
        help="Comma-separated geometric tolerances in Å, e.g. 0.001,0.005,0.01,0.02,0.05",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--model", default="mace-mp-0")
    parser.add_argument("--cache-mode", type=int, choices=[1, 2, 3, 4], default=3,
                        help="1=zeroth-order; 2=chunked dense Jacobian; 3=serial JVP/VJP; 4=batched JVP/VJP")
    parser.add_argument("--jacobian-chunk-size", type=int, default=64,
                        help="Edges per vectorized dense-Jacobian chunk when --cache-mode 2 is used")
    parser.add_argument("--batched-action-chunk-size", type=int, default=256,
                        help="Edges per batched JVP/VJP chunk when --cache-mode 4 is used")
    parser.add_argument("--output-dir", default="results/active/stage_b")
    parser.add_argument("--debug", action="store_true", help="Print concise Stage-B progress messages")
    parser.add_argument("--plan-only", action="store_true", help="Replay cache geometry and report sparse derivative dependencies only")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run the bounded accuracy pilot: step 40 with tolerances 0.01, 0.02, and 0.05 A.",
    )
    run(parser.parse_args())
