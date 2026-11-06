"""Reproducible accuracy/reuse/runtime benchmark for Frost paper experiments.

Runs exact MACE and edge-cache MD from identical deterministic initial
conditions. Expensive evaluations are intended for the project's GPU VM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin

from frost.e1.end_to_end_benchmark import IntegratedFrostCalculator
from frost.e1.force_error import compute_exact_forces
from frost.e1.model_interface import load_model
from frost.e1.systems import build_system
from frost.e1.trajectory import initialize_velocities


class TimedExactCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.elapsed = []

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        start = time.perf_counter()
        energy, forces = compute_exact_forces(atoms, self.model)
        self.elapsed.append(time.perf_counter() - start)
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prepare_production_state(model, args, seed):
    """Create one exact-equilibrated state shared by all production arms."""
    atoms = build_system(args.structure)
    atoms.calc = TimedExactCalculator(model)
    initialize_velocities(atoms, args.temperature, seed)
    np.random.seed(seed + 1000)
    dynamics = Langevin(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=args.friction_per_fs / units.fs,
    )
    dynamics.run(args.warmup_steps)
    sync()
    return atoms


def run_condition(model, method, args, seed, initial_atoms, tolerance=None):
    atoms = initial_atoms.copy()
    calc = TimedExactCalculator(model) if method == "exact" else IntegratedFrostCalculator(
        model, tolerance, cache_mode=args.cache_mode
    )
    if method != "exact" and args.cache_mode == 4:
        calc.batched_action_chunk_size = args.batched_action_chunk_size
    atoms.calc = calc
    np.random.seed(seed + 1000)
    dynamics = Langevin(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=args.friction_per_fs / units.fs,
    )

    if method != "exact":
        calc.begin_timing()

    rows = []
    for step in range(args.production_steps):
        sync()
        start = time.perf_counter()
        dynamics.run(1)
        sync()
        wall = time.perf_counter() - start
        snapshot = atoms.copy()
        exact_energy, exact_forces = compute_exact_forces(snapshot, model)
        if method == "exact":
            measured_energy = exact_energy
            measured_forces = exact_forces
        else:
            measured_energy = calc.last_frost_energy
            measured_forces = calc.last_frost_forces
        delta = np.asarray(measured_forces) - exact_forces
        force_norms = np.linalg.norm(delta, axis=1)
        rows.append({
            "method": method,
            "seed": seed,
            "tolerance_A": tolerance if tolerance is not None else "",
            "step": step,
            "wall_seconds": wall,
            "energy_error_eV": abs(float(measured_energy) - float(exact_energy)),
            "force_rms_meV_A": float(np.sqrt(np.mean(delta ** 2)) * 1000.0),
            "force_atom_p50_meV_A": float(np.percentile(force_norms, 50) * 1000.0),
            "force_atom_p95_meV_A": float(np.percentile(force_norms, 95) * 1000.0),
            "force_atom_p99_meV_A": float(np.percentile(force_norms, 99) * 1000.0),
            "force_atom_max_meV_A": float(np.max(force_norms) * 1000.0),
            "temperature_K": float(atoms.get_temperature()),
            "potential_energy_eV": float(atoms.get_potential_energy()),
            "finite": bool(np.isfinite(measured_energy) and np.isfinite(delta).all()),
        })
    return rows, calc


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["method"], row["tolerance_A"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (method, tolerance), group in groups.items():
        item = {"method": method, "tolerance_A": tolerance, "n_frames": len(group)}
        for column in (
            "wall_seconds", "energy_error_eV", "force_rms_meV_A",
            "force_atom_p95_meV_A", "force_atom_p99_meV_A", "force_atom_max_meV_A",
            "temperature_K",
        ):
            values = np.asarray([r[column] for r in group], dtype=float)
            item[column + "_mean"] = float(values.mean())
            item[column + "_median"] = float(np.median(values))
            item[column + "_p95"] = float(np.percentile(values, 95))
        item["all_finite"] = all(r["finite"] for r in group)
        summaries.append(item)
    return summaries


def run(args):
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model, device=args.device, dtype=args.dtype)
    all_rows = []
    cache_rows = []
    initial_state_digests = {}
    for seed in args.seeds:
        initial_atoms = prepare_production_state(model, args, seed)
        state_bytes = initial_atoms.get_positions().tobytes() + initial_atoms.get_momenta().tobytes()
        initial_state_digests[str(seed)] = hashlib.sha256(state_bytes).hexdigest()
        rows, _ = run_condition(model, "exact", args, seed, initial_atoms)
        all_rows.extend(rows)
        for tolerance in args.tolerances:
            rows, calc = run_condition(model, "cache", args, seed, initial_atoms, tolerance)
            all_rows.extend(rows)
            cache_rows.extend({"seed": seed, "tolerance_A": tolerance, **row} for row in calc.records)

    write_csv(out / "frame_metrics.csv", all_rows)
    write_csv(out / "cache_metrics.csv", cache_rows)
    manifest = {
        "experiment": "frost_paper_accuracy_reuse_runtime",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "model": args.model,
        "device": model.device,
        "dtype": args.dtype,
        "structure": args.structure,
        "temperature_K": args.temperature,
        "seeds": args.seeds,
        "initial_positions_and_momenta_sha256_by_seed": initial_state_digests,
        "warmup_steps": args.warmup_steps,
        "production_steps": args.production_steps,
        "timestep_fs": args.timestep_fs,
        "friction_per_fs": args.friction_per_fs,
        "tolerances_A": args.tolerances,
        "cache_mode": args.cache_mode,
        "warmup_protocol": "one exact-MACE equilibration per seed; post-equilibration positions and momenta shared by all production arms",
        "reference_refresh": "dirty edges refreshed immediately before cached evaluation",
        "frame_metrics_sha256": hashlib.sha256((out / "frame_metrics.csv").read_bytes()).hexdigest(),
        "summary": summarize(all_rows),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mace-mp-0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--structure", default="diamond_si")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 271])
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--production-steps", type=int, default=500)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--friction-per-fs", type=float, default=0.01)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--cache-mode", type=int, choices=(1, 3, 4), default=3,
                        help="1=zeroth-order; 3=serial JVP/VJP; 4=batched JVP/VJP")
    parser.add_argument("--batched-action-chunk-size", type=int, default=256)
    parser.add_argument("--output-dir", default="results/active/paper_baseline")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
