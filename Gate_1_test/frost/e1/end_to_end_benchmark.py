"""Short end-to-end exact-MACE versus integrated Frost benchmark.

This benchmark exercises the actual ASE calculator path with per-edge geometry
references and mode-3 JVP/VJP actions. It is deliberately short and does not
use dense Jacobians, Stage-B replay, or isolated E2 timings as a speed proxy.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import Trajectory
from ase.md.langevin import Langevin

from frost.e1.force_error import compute_cached_forces, compute_exact_forces, extract_layer1_local_inputs, make_layer1_edge_action
from frost.e1.geometry import build_edge_geometry_from_ase, compare_edge_key_sets, mace_edge_keys
from frost.e1.model_interface import load_model
from frost.e1.reference_cache import PerEdgeReferenceCache
from frost.e1.systems import build_system
from frost.e1.trajectory import initialize_velocities


class IntegratedFrostCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model, tolerance_A: float):
        super().__init__()
        self.model = model
        self.tolerance_A = tolerance_A
        self.cache = PerEdgeReferenceCache([tolerance_A])
        self.timestep = 0
        self.initialized = False
        self.timing_enabled = False
        self.records = []
        self.last_force_error_snapshot = None
        self.last_frost_energy = None
        self.last_frost_forces = None

    def begin_timing(self):
        self.timing_enabled = True

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        wall_start = time.perf_counter()
        current_atoms = atoms.copy()
        edges = build_edge_geometry_from_ase(current_atoms, self.model.cutoff)
        local = extract_layer1_local_inputs(self.model, current_atoms)
        current_vectors = local["vectors"].detach()
        current_g = local["mji"].detach()
        current_keys = mace_edge_keys(local["edge_index"], local["unit_shifts"])
        mapping = compare_edge_key_sets(edges.keys(), current_keys)
        if not mapping["matched"]:
            raise RuntimeError(f"Frost edge mapping mismatch: {mapping}")
        current_by_key = {key: index for index, key in enumerate(current_keys)}

        if not self.initialized:
            self.cache.initialize(edges, timestep=self.timestep)
            self.initialized = True
        stats = self.cache.step(edges, timestep=self.timestep)
        dirty_keys = stats["dirty_keys_by_eps"][self.tolerance_A]
        clean_mask = torch.zeros(len(current_keys), dtype=torch.bool)
        g_reference = current_g.clone()
        vectors_reference = current_vectors.clone()
        action_fns = [None] * len(current_keys)

        for key, index in current_by_key.items():
            entry = self.cache._entries[key]
            if key in dirty_keys or entry.g_ref is None:
                entry.g_ref = current_g[index].clone()
                entry.vectors_ref = current_vectors[index].clone()
                entry.action = make_layer1_edge_action(self.model, local, index)
            else:
                clean_mask[index] = True
            g_reference[index] = entry.g_ref
            vectors_reference[index] = entry.vectors_ref
            action_fns[index] = entry.action

        original_calc = current_atoms.calc
        try:
            frost_energy, frost_forces = compute_cached_forces(
                current_atoms,
                self.model,
                clean_mask,
                g_reference,
                mode=3,
                vectors=current_vectors,
                vectors_ref=vectors_reference,
                action_fns=action_fns,
            )
        finally:
            current_atoms.calc = original_calc

        elapsed = time.perf_counter() - wall_start
        self.last_frost_energy = float(frost_energy)
        self.last_frost_forces = np.asarray(frost_forces).copy()
        self.results = {
            "energy": float(frost_energy),
            "free_energy": float(frost_energy),
            "forces": self.last_frost_forces,
        }
        if self.timing_enabled:
            self.records.append({
                "step": self.timestep,
                "wall_seconds": elapsed,
                "n_edges": stats["n_edges_current"],
                "n_clean": int(clean_mask.sum().item()),
                "n_dirty": int((~clean_mask).sum().item()),
                "phi": float(stats["dirty_fractions"][self.tolerance_A]),
                "clean_fraction": float(clean_mask.float().mean().item()),
                "n_new": stats["n_edges_new"],
                "n_removed": stats["n_edges_removed"],
                "reference_action_edges": int(clean_mask.sum().item()),
                "action_mode": 3,
                "dense_jacobian_enabled": False,
            })
        self.timestep += 1


class ExactCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model):
        super().__init__()
        self.model = model

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy, forces = compute_exact_forces(atoms, self.model)
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}


def run_trajectory(model, method, temperature, warmup_steps, timed_steps, seed, tolerance_A=None):
    atoms = build_system("diamond_si")
    if method == "exact":
        calculator = ExactCalculator(model)
    else:
        calculator = IntegratedFrostCalculator(model, tolerance_A)
    atoms.calc = calculator
    initialize_velocities(atoms, temperature, seed)
    dynamics = Langevin(
        atoms,
        timestep=1.0 * units.fs,
        temperature_K=temperature,
        friction=0.01 / units.fs,
    )
    np.random.seed(seed + 1000)
    warmup_start = time.perf_counter()
    dynamics.run(warmup_steps)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_start

    if method == "frost":
        calculator.begin_timing()
    step_rows = []
    accuracy_rows = []
    trajectory_start = time.perf_counter()
    for timed_step in range(timed_steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.perf_counter()
        dynamics.run(1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_rows.append({
            "step": timed_step,
            "wall_seconds": time.perf_counter() - step_start,
            "temperature_K": float(atoms.get_temperature()),
            "energy_eV": float(atoms.get_potential_energy()),
        })
        if method == "frost":
            snapshot = atoms.copy()
            exact_energy, exact_forces = compute_exact_forces(snapshot, model)
            delta = calculator.last_frost_forces - exact_forces
            accuracy_rows.append({
                "step": timed_step,
                "energy_error_eV": abs(calculator.last_frost_energy - exact_energy),
                "force_rms_error_meV_A": float(np.sqrt(np.mean(delta ** 2)) * 1000.0),
                "force_max_error_meV_A": float(np.abs(delta).max() * 1000.0),
                "finite": bool(np.isfinite(exact_energy) and np.isfinite(calculator.last_frost_energy) and np.isfinite(delta).all()),
            })
    total_seconds = time.perf_counter() - trajectory_start

    return {
        "method": method,
        "warmup_seconds": warmup_seconds,
        "timed_seconds": total_seconds,
        "step_rows": step_rows,
        "frost_cache_rows": calculator.records if method == "frost" else [],
        "accuracy_rows": accuracy_rows,
        "final_temperature_K": float(atoms.get_temperature()),
        "final_energy_eV": float(atoms.get_potential_energy()),
    }


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "mean_seconds": float(np.mean(values)),
        "median_seconds": float(np.median(values)),
        "p90_seconds": float(np.percentile(values, 90)),
        "min_seconds": float(np.min(values)),
        "max_seconds": float(np.max(values)),
    }


def run(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model = load_model("mace-mp-0", device="cuda", dtype="float64")
    exact = run_trajectory(model, "exact", args.temperature, args.warmup_steps, args.timed_steps, args.seed)
    frost = run_trajectory(model, "frost", args.temperature, args.warmup_steps, args.timed_steps, args.seed, args.tolerance_A)

    exact_times = [row["wall_seconds"] for row in exact["step_rows"]]
    frost_times = [row["wall_seconds"] for row in frost["step_rows"]]
    summary = {
        "experiment": "short_end_to_end_exact_vs_integrated_frost",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model": "mace-mp-0",
        "device": model.device,
        "dtype": "float64",
        "system": "diamond_si_64_atoms",
        "temperature_K": args.temperature,
        "seed": args.seed,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "tolerance_A": args.tolerance_A,
        "dense_jacobian_enabled": False,
        "frost_action_mode": 3,
        "frost_action_path_used": True,
        "exact": {
            "warmup_seconds": exact["warmup_seconds"],
            "timed_total_seconds": exact["timed_seconds"],
            "step_timing": stats(exact_times),
            "throughput_steps_per_second": args.timed_steps / exact["timed_seconds"],
        },
        "frost": {
            "warmup_seconds": frost["warmup_seconds"],
            "timed_total_seconds": frost["timed_seconds"],
            "step_timing": stats(frost_times),
            "throughput_steps_per_second": args.timed_steps / frost["timed_seconds"],
        },
        "speedup": {
            "total_wall_speedup": exact["timed_seconds"] / frost["timed_seconds"],
            "median_step_speedup": np.median(exact_times) / np.median(frost_times),
        },
        "accuracy": {
            "rows": frost["accuracy_rows"],
            "all_finite": bool(all(row["finite"] for row in frost["accuracy_rows"])),
        },
        "reuse": {
            "mean_phi": float(np.mean([row["phi"] for row in frost["frost_cache_rows"]])),
            "mean_clean_fraction": float(np.mean([row["clean_fraction"] for row in frost["frost_cache_rows"]])),
            "mean_n_edges": float(np.mean([row["n_edges"] for row in frost["frost_cache_rows"]])),
        },
        "stability": {
            "exact_final_temperature_K": exact["final_temperature_K"],
            "frost_final_temperature_K": frost["final_temperature_K"],
            "exact_final_energy_eV": exact["final_energy_eV"],
            "frost_final_energy_eV": frost["final_energy_eV"],
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output / "per_step_timing.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "step", "wall_seconds", "temperature_K", "energy_eV"])
        writer.writeheader()
        for result in (exact, frost):
            for row in result["step_rows"]:
                writer.writerow({"method": result["method"], **row})
    with (output / "frost_cache.csv").open("w", newline="") as handle:
        rows = frost["frost_cache_rows"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "force_errors.csv").open("w", newline="") as handle:
        rows = frost["accuracy_rows"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output / "README.md").write_text(
        "# Frost end-to-end benchmark\n\n"
        "This short benchmark compares exact MACE and the integrated mode-3 Frost calculator. "
        "Timed steps include the complete ASE MD step and all Frost cache/action work. "
        "Dense Jacobians and Stage-B replay are disabled.\n\n"
        f"Command: `python -m frost.e1.end_to_end_benchmark --output-dir {output}`\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--tolerance-A", type=float, default=0.01)
    parser.add_argument("--output-dir", default="results/e2/end_to_end_frost_short")
    run(parser.parse_args())
