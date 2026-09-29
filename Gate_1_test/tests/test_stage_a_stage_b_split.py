import csv
import json
from pathlib import Path

import numpy as np
from ase import Atoms

from frost.e1 import temporal_redundancy_experiment as stage_a_mod
from frost.e1 import stage_b_accuracy_sampling as stage_b_mod


class DummyModel:
    cutoff = 2.0


def _write_stage_a_csv(path: Path, n_steps: int, tolerances):
    rows = []
    for step in range(n_steps):
        for tol in tolerances:
            rows.append(
                {
                    "step": step,
                    "tolerance_A": tol,
                    "n_edges": 12,
                    "n_clean": 10,
                    "n_dirty": 2,
                    "n_new": 0,
                    "n_removed": 0,
                    "n_refreshed": 0,
                    "phi": 0.2,
                    "clean_fraction": 0.8,
                    "mean_delta_r": 0.01,
                    "median_delta_r": 0.01,
                    "p90_delta_r": 0.02,
                    "p95_delta_r": 0.03,
                    "p99_delta_r": 0.04,
                    "max_delta_r": 0.05,
                    "generation_max": 1,
                }
            )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_stage_a_no_sampling_and_no_dense_jacobian(monkeypatch, tmp_path):
    calls = {"md": 0}

    def fake_run_md_trajectory(*args, **kwargs):
        calls["md"] += 1
        step_callback = kwargs["step_callback"]
        for step in range(5):
            atoms = Atoms("Si2", positions=[[0, 0, 0], [1.1, 0, 0]], cell=[10, 10, 10], pbc=True)
            step_callback(atoms, step)
        return {"n_atoms": 2, "n_production_steps": 5}

    monkeypatch.setattr(stage_a_mod, "load_model", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(stage_a_mod, "build_system", lambda *args, **kwargs: Atoms("Si2", positions=[[0, 0, 0], [1.1, 0, 0]], cell=[10, 10, 10], pbc=True))
    monkeypatch.setattr(stage_a_mod, "run_md_trajectory", fake_run_md_trajectory)

    output_dir = tmp_path / "stageA"
    out = stage_a_mod.run(
        type(
            "Args",
            (),
            {
                "stage_a": True,
                "equilibration_steps": 1,
                "production_steps": 5,
                "seed": 42,
                "output_dir": str(output_dir),
                "tolerances": [0.001, 0.005, 0.01],
                "debug": False,
                "sample_interval": 999999,
            },
        )()
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["stage"] == "A"
    assert summary["jacobian_enabled"] is False
    assert summary["sampled_steps"] == []
    assert summary["sampled_dense_jacobian_seconds"] == 0.0
    assert calls["md"] == 1
    assert (output_dir / "trajectory.traj").exists()
    assert (output_dir / "per_timestep.csv").exists()


def test_stage_b_selection_and_csv_validation(monkeypatch, tmp_path):
    input_csv = tmp_path / "per_timestep.csv"
    traj_path = tmp_path / "trajectory.traj"
    _write_stage_a_csv(input_csv, n_steps=5, tolerances=[0.001, 0.005, 0.01])

    atoms_list = [Atoms("Si2", positions=[[0, 0, 0], [1.1, 0, 0]], cell=[10, 10, 10], pbc=True) for _ in range(5)]
    from ase.io import Trajectory
    with Trajectory(str(traj_path), "w") as traj:
        for atoms in atoms_list:
            traj.write(atoms)

    output_dir = tmp_path / "stageB"
    monkeypatch.setattr(stage_b_mod, "load_model", lambda *args, **kwargs: DummyModel())
    stage_b_mod.run(
        type(
            "Args",
            (),
            {
                "input": str(input_csv),
                "trajectory": str(traj_path),
                "steps": "1,3",
                "tolerances": "0.001,0.01",
                "seed": 42,
                "device": "cpu",
                "debug": False,
                "pilot": False,
                "plan_only": True,
                "output_dir": str(output_dir),
            },
        )()
    )

    summary = json.loads((output_dir / "plan.json").read_text())
    assert summary["stage"] == "B"
    assert summary["unique_reference_frames"] == 1
    assert summary["selected_rows"] == 4
