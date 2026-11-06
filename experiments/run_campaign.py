"""Run Stage A, gate on the W0 audit, then execute sparse Stage B."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text())
    root = Path(config["output_root"])
    if not root.is_absolute():
        root = ROOT / root
    sizes = config.get("system_sizes_atoms", [None])
    for structure in config["structures"]:
        for temperature in config["temperatures_K"]:
            for system_size in sizes:
                for seed in config["seeds"]:
                    size_tag = f"_N{system_size}" if system_size else ""
                    tag = f"{structure}{size_tag}_T{temperature:g}_seed{seed}"
                    stage_a = root / tag / "stage_a"
                    base = [
                        sys.executable, "-m", "frost.e1.temporal_redundancy_experiment",
                        "--stage-a", "--structure", structure, "--model", config["model"],
                        "--device", config["device"], "--dtype", config["dtype"],
                        "--temperature", str(temperature), "--seed", str(seed),
                        "--equilibration-steps", str(config["equilibration_steps"]),
                        "--production-steps", str(config["production_steps"]),
                        "--timestep-fs", str(config["timestep_fs"]),
                        "--friction-per-fs", str(config["friction_per_fs"]),
                        "--tolerances", *map(str, config["tolerances_A"]),
                        "--output-dir", str(stage_a),
                    ]
                    if system_size:
                        base.extend(["--system-size", str(system_size)])

                    if args.stage in ("all", "stage-a"):
                        print("RUN", subprocess.list2cmdline(base), flush=True)
                        subprocess.run(base, cwd=ROOT, check=True)

                    case_dir = root / tag
                    audit_dir = case_dir / "w0_audit"
                    if args.stage in ("all", "audit"):
                        if args.stage == "audit" and audit_dir.exists() and any(audit_dir.iterdir()):
                            revision = 2
                            while (case_dir / f"w0_audit_r{revision}").exists():
                                revision += 1
                            audit_dir = case_dir / f"w0_audit_r{revision}"
                        audit_cmd = [
                            sys.executable, "-m", "frost.e1.e3_audit",
                            "--trajectory", str(stage_a / "trajectory.traj"),
                            "--step", "0", "--tolerances", "0.005", "0.01",
                            "--model", config["model"], "--device", config["device"],
                            "--dtype", config["dtype"], "--output-dir", str(audit_dir),
                        ]
                        print("RUN", subprocess.list2cmdline(audit_cmd), flush=True)
                        subprocess.run(audit_cmd, cwd=ROOT, check=True)

                    if args.stage in ("all", "stage-b"):
                        audit_candidates = [case_dir / "w0_audit", *sorted(case_dir.glob("w0_audit_r*"))]
                        passed = [candidate for candidate in audit_candidates
                                  if (candidate / "summary.json").exists()
                                  and json.loads((candidate / "summary.json").read_text()).get("audit_passed", False)]
                        if not passed:
                            raise RuntimeError(f"W0 audit is missing or failed for {tag}; stop before Stage B")
                        for mode in config["accuracy_cache_modes"]:
                            target = root / tag / f"stage_b_mode{mode}"
                            stage_b_cmd = [
                                sys.executable, "-m", "frost.e1.stage_b_accuracy_sampling",
                                "--input", str(stage_a / "per_timestep.csv"),
                                "--trajectory", str(stage_a / "trajectory.traj"),
                                "--sample-count", str(config["accuracy_sample_count"]),
                                "--tolerances", ",".join(map(str, config["tolerances_A"])),
                                "--model", config["model"], "--device", config["device"],
                                "--dtype", config["dtype"], "--cache-mode", str(mode),
                                "--output-dir", str(target),
                            ]
                            if mode == 2:
                                stage_b_cmd.extend([
                                    "--jacobian-chunk-size",
                                    str(config.get("jacobian_chunk_size", 64)),
                                ])
                            elif mode == 4:
                                stage_b_cmd.extend([
                                    "--batched-action-chunk-size",
                                    str(config.get("batched_action_chunk_size", 256)),
                                ])
                            print("RUN", subprocess.list2cmdline(stage_b_cmd), flush=True)
                            subprocess.run(stage_b_cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/generalization.json")
    parser.add_argument("--stage", choices=["all", "stage-a", "audit", "stage-b"], default="all")
    run(parser.parse_args())
