"""W0 cache-path audit and controlled single-atom perturbation experiment."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from frost.e1.end_to_end_benchmark import IntegratedFrostCalculator
from frost.e1.force_error import compute_exact_forces
from frost.e1.model_interface import load_model
from frost.e1.metadata import capture_environment


def run(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frames = read(args.trajectory, index=":")
    frame = frames[args.step]
    model = load_model(args.model, device=args.device, dtype=args.dtype)
    started = time.time()
    rows = []
    edge_rows = []
    controls = [("exact_mode_path_control", 0, "disabled"), ("reuse_disabled", 1, "disabled"),
                ("zeroth_threshold", 1, "threshold"), ("first_threshold", 3, "threshold"),
                ("batched_first_threshold", 4, "threshold"),
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
                    edge_displacements = audit.get(
                        "edge_displacement_A", np.full(len(msg_error), np.nan)
                    )
                    edge_ages = audit.get(
                        "reference_age_steps", np.full(len(msg_error), np.nan)
                    )
                    for edge_index, error in enumerate(msg_error):
                        edge_rows.append({
                            "step": args.step,
                            "tolerance_A": tolerance,
                            "arm": arm,
                            "displacement_A": displacement,
                            "edge_index": edge_index,
                            "edge_displacement_A": float(edge_displacements[edge_index]),
                            "reference_age_steps": float(edge_ages[edge_index]),
                            "clean": bool(clean[edge_index]),
                            "message_error": float(error),
                        })
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
    zero_rows = [r for r in rows if r["displacement_A"] == 0]
    force_control_arms = {"exact_mode_path_control", "reuse_disabled", "first_threshold", "batched_first_threshold"}
    validated_zero_rows = [r for r in zero_rows if r["arm"] in force_control_arms]
    zeroth_rows = [r for r in zero_rows if r["arm"] in {"zeroth_threshold", "all_message_reuse_diagnostic"}]
    checks = [
        ("all_finite", summary_all_finite := all(r["all_finite"] for r in rows), True),
        ("storage_independence", summary_storage := all(r["message_storage_separate"] for r in rows), True),
    ]
    summary = {"experiment": "frost_w0_cache_path_audit", "test_name": "W0 integrated cache-path audit",
               "trajectory": str(args.trajectory),
               "step": args.step, "tolerances_A": args.tolerances,
               "displacements_A": displacements, "n_rows": len(rows),
               "n_edge_rows": len(edge_rows),
               "all_finite": summary_all_finite,
               "message_storage_separate": summary_storage,
               "zero_displacement_max_energy_error_eV": max(r["energy_error_eV"] for r in zero_rows),
               "zero_displacement_force_validation_arms": sorted(force_control_arms),
               "zero_displacement_max_force_rms_meV_A": max(r["force_rms_meV_A"] for r in validated_zero_rows),
               "zeroth_order_zero_displacement_force_rms_meV_A": max(r["force_rms_meV_A"] for r in zeroth_rows),
               "dirty_message_max_abs_error": max((r["message_dirty_max"] for r in rows if r["n_dirty"] > 0), default=0.0)}
    summary["audit_passed"] = bool(summary["all_finite"] and summary["message_storage_separate"]
        and summary["zero_displacement_max_energy_error_eV"] < 1e-8
        and summary["zero_displacement_max_force_rms_meV_A"] < 1e-5
        and summary["dirty_message_max_abs_error"] < 1e-10)
    checks.extend([
        ("zero_displacement_energy", summary["zero_displacement_max_energy_error_eV"] < 1e-8, 1e-8),
        ("zero_displacement_force", summary["zero_displacement_max_force_rms_meV_A"] < 1e-5, 1e-5),
        ("dirty_message_exactness", summary["dirty_message_max_abs_error"] < 1e-10, 1e-10),
    ])
    with (output / "audit_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "criterion"])
        writer.writeheader()
        writer.writerows({"check": name, "passed": passed, "criterion": criterion}
                         for name, passed, criterion in checks)
    with (output / "message_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["step", "tolerance_A", "arm", "displacement_A", "edge_index",
                  "edge_displacement_A", "reference_age_steps", "clean",
                  "message_error"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(edge_rows)
    with (output / "force_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["step", "tolerance_A", "arm", "displacement_A",
                  "energy_error_eV", "force_rms_meV_A", "force_max_meV_A", "all_finite"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    summary.update({
        "pass": summary["audit_passed"],
        "numerical_tolerances": {
            "energy_eV": 1e-8, "force_rms_meV_A": 1e-5, "dirty_message": 1e-10,
        },
        "failure_reason": None if summary["audit_passed"] else "One or more mandatory W0 checks failed",
        "configuration": vars(args),
        "software": capture_environment(Path(__file__).resolve().parents[2]),
        "execution_environment": {
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "started_unix": started,
            "duration_seconds": time.time() - started,
        },
    })
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    (output / "metadata.json").write_text(json.dumps({
        "experiment": summary["experiment"], "test_name": summary["test_name"],
        "configuration": vars(args), "software": summary["software"],
        "execution_environment": summary["execution_environment"],
    }, indent=2))
    (output / "audit.log").write_text(
        f"passed={summary['audit_passed']}\nrows={len(rows)}\n"
        f"failure_reason={summary['failure_reason']}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not summary["audit_passed"]:
        raise SystemExit(1)


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
