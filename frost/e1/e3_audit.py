"""W0 cache-path audit and controlled single-atom perturbation experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from frost.e1.end_to_end_benchmark import IntegratedFrostCalculator
from frost.e1.force_error import compute_exact_forces
from frost.e1.model_interface import load_model


def run(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frames = read(args.trajectory, index=":")
    frame = frames[args.step]
    model = load_model(args.model, device=args.device, dtype=args.dtype)
    rows = []
    controls = [("exact_mode_path_control", 0, "disabled"), ("reuse_disabled", 1, "disabled"),
                ("zeroth_threshold", 1, "threshold"), ("first_threshold", 3, "threshold"),
                ("all_message_reuse_diagnostic", 1, "all_clean")]
    displacements = [0.0, *args.displacements]
    for tolerance in args.tolerances:
        for arm, mode, policy in controls:
            for displacement in displacements:
                atoms = frame.copy()
                calc = IntegratedFrostCalculator(model, tolerance, cache_mode=mode,
                                                  cache_policy=policy, audit_capture=True)
                atoms.calc = calc
                # Establish reference messages at the unperturbed coordinates.
                atoms.get_forces()
                if displacement:
                    pos = atoms.get_positions()
                    pos[args.atom, args.axis] += displacement
                    atoms.set_positions(pos)
                exact_atoms = atoms.copy()
                energy, exact_forces = compute_exact_forces(exact_atoms, model)
                # Force ASE to evaluate the current coordinates through Frost.
                atoms.calc.results = {}
                forces = atoms.get_forces()
                frost_energy = float(calc.last_frost_energy)
                delta_f = forces - exact_forces
                audit = calc.last_message_audit or {}
                exact_msg = audit.get("mji_exact")
                used_msg = audit.get("mji_used")
                clean = audit.get("clean_mask", np.zeros(0, dtype=bool))
                if exact_msg is not None and used_msg is not None and len(clean) == len(exact_msg):
                    msg_error = np.linalg.norm(used_msg - exact_msg, axis=1)
                    clean_rms = float(np.sqrt(np.mean(msg_error[clean] ** 2))) if clean.any() else 0.0
                    dirty_rms = float(np.sqrt(np.mean(msg_error[~clean] ** 2))) if (~clean).any() else 0.0
                    clean_max = float(msg_error[clean].max()) if clean.any() else 0.0
                    dirty_max = float(msg_error[~clean].max()) if (~clean).any() else 0.0
                else:
                    clean_rms = dirty_rms = clean_max = dirty_max = float("nan")
                rows.append({
                    "step": args.step, "tolerance_A": tolerance, "arm": arm,
                    "cache_mode": mode, "cache_policy": policy,
                    "displacement_A": displacement, "n_clean": int(clean.sum()),
                    "n_dirty": int(len(clean) - clean.sum()),
                    "message_clean_rms": clean_rms, "message_clean_max": clean_max,
                    "message_dirty_rms": dirty_rms, "message_dirty_max": dirty_max,
                    "energy_error_eV": abs(float(frost_energy) - float(energy)),
                    "force_rms_meV_A": float(np.sqrt(np.mean(delta_f ** 2)) * 1000),
                    "force_max_meV_A": float(np.linalg.norm(delta_f, axis=1).max() * 1000),
                    "all_finite": bool(np.isfinite(forces).all() and np.isfinite(exact_forces).all()),
                    "message_storage_separate": bool(audit.get("separate_storage", False)),
                })
    fields = list(rows[0])
    with (output / "audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = {"experiment": "frost_w0_cache_path_audit", "trajectory": str(args.trajectory),
               "step": args.step, "tolerances_A": args.tolerances,
               "displacements_A": displacements, "n_rows": len(rows),
               "all_finite": all(r["all_finite"] for r in rows),
               "message_storage_separate": all(r["message_storage_separate"] for r in rows),
               "zero_displacement_max_energy_error_eV": max(r["energy_error_eV"] for r in rows if r["displacement_A"] == 0),
               "zero_displacement_max_force_rms_meV_A": max(r["force_rms_meV_A"] for r in rows if r["displacement_A"] == 0),
               "dirty_message_max_abs_error": max((r["message_dirty_max"] for r in rows if r["n_dirty"] > 0), default=0.0)}
    summary["audit_passed"] = bool(summary["all_finite"] and summary["message_storage_separate"]
        and summary["zero_displacement_max_energy_error_eV"] < 1e-8
        and summary["zero_displacement_max_force_rms_meV_A"] < 1e-5
        and summary["dirty_message_max_abs_error"] < 1e-10)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory", required=True)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--tolerances", type=float, nargs="+", default=[0.005, 0.01])
    p.add_argument("--displacements", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 5e-2])
    p.add_argument("--atom", type=int, default=0)
    p.add_argument("--axis", type=int, choices=[0, 1, 2], default=0)
    p.add_argument("--model", default="mace-mp-0")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--output-dir", default="results/active/w0_audit")
    run(p.parse_args())
