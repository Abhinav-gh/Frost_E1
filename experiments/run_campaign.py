"""Run a validated Stage-A -> W0 -> Stage-B campaign.

The runner never overwrites a non-empty stage directory.  ``--resume`` only
skips a stage with a completion marker and a valid summary; partial output is
reported as an error so an interrupted experiment cannot be mistaken for
evidence.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from frost.e1.metadata import capture_environment

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONFIG_KEYS = {
    "experiment_name", "model", "device", "dtype", "structures", "temperatures_K", "seeds",
    "equilibration_steps", "production_steps", "timestep_fs",
    "friction_per_fs", "tolerances_A", "accuracy_sample_count",
    "accuracy_cache_modes", "sampling", "cache_policy",
    "reference_update_policy", "jvp_vjp_mode", "output_root",
}


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_CONFIG_KEYS - set(config)
    if missing:
        raise ValueError(f"Campaign config is missing: {', '.join(sorted(missing))}")
    if config["dtype"] not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")
    if not config["structures"] or not config["temperatures_K"] or not config["seeds"]:
        raise ValueError("structures, temperatures_K, and seeds must be non-empty")
    if any(mode not in {1, 2, 3, 4} for mode in config["accuracy_cache_modes"]):
        raise ValueError("accuracy_cache_modes must contain only 1, 2, 3, or 4")
    if any(value <= 0 for value in config["tolerances_A"]):
        raise ValueError("tolerances_A must be positive")
    if config["accuracy_sample_count"] < 1:
        raise ValueError("accuracy_sample_count must be positive")
    return config


def _complete(directory: Path, summary_name: str) -> bool:
    return (
        (directory / ".complete").is_file()
        and (directory / summary_name).is_file()
    )


def _mark_complete(directory: Path) -> None:
    (directory / ".complete").write_text("complete\n", encoding="utf-8")


def _run(command: list[str], *, log_path: Path) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {subprocess.list2cmdline(command)}\n")
        completed = subprocess.run(
            command, cwd=ROOT, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        log.write(completed.stdout)
        print(completed.stdout, end="", flush=True)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def _write_manifest(path: Path, config_path: Path, config: dict) -> None:
    payload = {
        "config_path": str(config_path),
        "config": config,
        "environment": capture_environment(ROOT),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "command": " ".join(sys.argv),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _finish_manifest(path: Path, status: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = status
    payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(args):
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = _load_config(config_path)
    root = Path(config["output_root"])
    if not root.is_absolute():
        root = ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "campaign_manifest.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(
            f"{manifest_path} exists; use --resume to continue this campaign "
            "or choose a new output_root"
        )
    if not manifest_path.exists():
        _write_manifest(manifest_path, config_path, config)
    elif json.loads(manifest_path.read_text(encoding="utf-8"))["config"] != config:
        raise ValueError("Cannot resume with a configuration different from campaign_manifest.json")
    log_path = root / "run.log"
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
                        if args.resume and _complete(stage_a, "summary.json"):
                            print("SKIP complete Stage A", stage_a, flush=True)
                        elif stage_a.exists() and any(stage_a.iterdir()):
                            raise FileExistsError(
                                f"Stage-A output is non-empty but incomplete: {stage_a}"
                            )
                        else:
                            _run(base, log_path=log_path)
                            _mark_complete(stage_a)

                    case_dir = root / tag
                    audit_dir = case_dir / "w0_audit"
                    if args.stage in ("all", "audit", "stage-b"):
                        if args.resume and _complete(audit_dir, "summary.json"):
                            print("SKIP complete W0 audit", audit_dir, flush=True)
                        else:
                            if audit_dir.exists() and any(audit_dir.iterdir()):
                                raise FileExistsError(
                                    f"W0 output is non-empty but incomplete: {audit_dir}"
                                )
                            audit_cmd = [
                                sys.executable, "-m", "frost.e1.e3_audit",
                                "--trajectory", str(stage_a / "trajectory.traj"),
                                "--step", "0", "--tolerances", "0.005", "0.01",
                                "--model", config["model"], "--device", config["device"],
                                "--dtype", config["dtype"], "--output-dir", str(audit_dir),
                            ]
                            _run(audit_cmd, log_path=log_path)
                            _mark_complete(audit_dir)

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
                            if args.resume and _complete(target, "summary.json"):
                                print("SKIP complete Stage B", target, flush=True)
                                continue
                            if target.exists() and any(target.iterdir()):
                                raise FileExistsError(
                                    f"Stage-B output is non-empty but incomplete: {target}"
                                )
                            _run(stage_b_cmd, log_path=log_path)
                            _mark_complete(target)
    _finish_manifest(manifest_path, "completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/generalization.json")
    parser.add_argument("--stage", choices=["all", "stage-a", "audit", "stage-b"], default="all")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip only stages with .complete and summary.json markers",
    )
    options = parser.parse_args()
    try:
        run(options)
    except Exception:
        try:
            config_path = Path(options.config)
            if not config_path.is_absolute():
                config_path = ROOT / config_path
            config = json.loads(config_path.read_text(encoding="utf-8"))
            manifest = Path(config["output_root"])
            if not manifest.is_absolute():
                manifest = ROOT / manifest
            manifest /= "campaign_manifest.json"
            if manifest.exists():
                _finish_manifest(manifest, "failed")
        finally:
            raise
