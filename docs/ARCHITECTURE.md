# Frost architecture

## Layers

- **Core:** [`geometry.py`](../frost/e1/geometry.py), [`reference_cache.py`](../frost/e1/reference_cache.py), [`mace_patch.py`](../frost/e1/mace_patch.py), and [`force_error.py`](../frost/e1/force_error.py).
- **Model boundary:** [`model_interface.py`](../frost/e1/model_interface.py) owns MACE loading, cutoff discovery, and the ASE calculator.
- **Experiment runners:** [`temporal_redundancy_experiment.py`](../frost/e1/temporal_redundancy_experiment.py) (Stage A), [`stage_b_accuracy_sampling.py`](../frost/e1/stage_b_accuracy_sampling.py) (Stage B), [`e3_audit.py`](../frost/e1/e3_audit.py) (W0), [`inference_benchmark.py`](../frost/e1/inference_benchmark.py) (W4), and [`paper_benchmark.py`](../frost/e1/paper_benchmark.py) (W5).
- **Analysis:** [`frontier_analysis.py`](../frost/e1/frontier_analysis.py), [`edge_error_analysis.py`](../frost/e1/edge_error_analysis.py), and [`cost_model_analysis.py`](../frost/e1/cost_model_analysis.py).
- **Orchestration:** [`run_campaign.py`](../experiments/run_campaign.py) validates a JSON campaign and invokes stages as separate processes.

## Data flow

Stage A runs exact MACE MD, writes every production frame to
`trajectory.traj`, and updates only geometry-only per-edge references. Its
`per_timestep.csv` is the input to Stage B. Stage B replays the reference
state, computes exact forces and selected Frost approximations, and writes
`accuracy_results.csv` plus `summary.json`. W0 is run on a saved Stage-A
trajectory and must pass before Stage B in an `all` campaign.

## Cache lifecycle

Each canonical directed edge has reference geometry, reference timestep, age,
and optional message/action state. New or dirty edges are refreshed before
substitution. Clean edges use cached zeroth-order messages or a first-order
JVP/VJP action. Edge-key mismatches raise an error; no positional ordering
assumption is used.

## Output lifecycle

Every runner refuses a non-empty output directory. A campaign case is
`<output_root>/<structure>_T<temperature>_seed<seed>/` with `stage_a`,
`w0_audit`, and one `stage_b_mode<N>` directory. Historical outputs remain in
`results/legacy_baseline/`; new outputs belong under `results/active/`.
