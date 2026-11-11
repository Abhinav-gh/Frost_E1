# Frost: temporal edge-message caching in MACE

Frost measures whether edge-local MACE message reuse is geometrically common,
numerically accurate, and beneficial in complete inference/MD execution. A
speedup is not assumed. Stage A measures geometry only; Stage B measures
stale-cache error against exact MACE; W4/E4 measures fixed-configuration
complete-calculator cost; W5/E5 measures matched-start MD rollout behavior.

## Repository

- [`frost/e1/`](frost/e1): geometry, reference cache, MACE integration,
  JVP/VJP actions, W0 audit, Stage A/B, W4/E4, and W5/E5.
- [`experiments/run_campaign.py`](experiments/run_campaign.py): validated
  Stage A -> W0 -> Stage B campaign runner with resume protection.
- [`analysis/`](analysis): raw-result aggregation and figure generation.
- [`configs/`](configs): smoke, pilot, generalization, temperature, scale,
  W4/E4, and W5/E5 configurations.
- [`results/legacy_baseline/`](results/legacy_baseline): preserved historical
  artifacts. Never overwrite.
- [`results/active/`](results/active): new unique experiment outputs.
- [`docs/EXPERIMENT_ROADMAP.md`](docs/EXPERIMENT_ROADMAP.md): scientific gates
  and claim boundaries.

See [`docs/IMPLEMENTATION_REPORT.md`](docs/IMPLEMENTATION_REPORT.md),
[`docs/RESULTS_SCHEMA.md`](docs/RESULTS_SCHEMA.md), and
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

## Installation on the GPU VM

Do not run production experiments on a workstation or without checking the
CUDA/PyTorch compatibility of the VM.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the PyTorch wheel appropriate for the VM's CUDA driver.
python -m pip install mace-torch
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

MACE-MP-0 may download its checkpoint on first use. Ensure the VM can access
the checkpoint or has a populated local cache.

## Execution order

Every command below is intended to run from the repository root. Each output
directory must be new or be resumed with the exact same configuration.

### P0. CPU checks and GPU smoke test

The unit suite is fast and does not run MD:

```bash
python -m pytest -q
python -m compileall -q frost experiments analysis
```

The smoke campaign is non-publication validation only:

```bash
python experiments/run_campaign.py \
  --config configs/smoke.json --stage all
```

The campaign runs Stage A, then W0, and stops before Stage B if W0 fails.
Inspect:

```bash
cat results/active/smoke/diamond_si_T300_seed42/w0_audit/audit_summary.json
cat results/active/smoke/diamond_si_T300_seed42/w0_audit/audit_checks.csv
```

### P0. W0 scientific correctness audit

W0 compares exact, disabled, all-dirty, zeroth-order, serial first-order,
batched first-order, and all-clean diagnostic arms. It writes
`audit_summary.json`, `audit_checks.csv`, `message_errors.csv`,
`force_errors.csv`, `audit.log`, and `metadata.json`.

Run it on a saved Stage-A trajectory:

```bash
python -m frost.e1.e3_audit \
  --trajectory results/active/smoke/diamond_si_T300_seed42/stage_a/trajectory.traj \
  --step 0 --tolerances 0.005 0.01 \
  --output-dir results/active/w0_smoke
```

The command returns nonzero on mandatory failure. Do not interpret Stage B
results if `audit_summary.json` has `"audit_passed": false`.

### P1. Core reuse/accuracy frontier

Run the pilot first:

```bash
python experiments/run_campaign.py \
  --config configs/pilot.json --stage all
```

Then run the required 50-snapshot, seven-tolerance, three-seed campaign:

```bash
python experiments/run_campaign.py \
  --config configs/generalization.json --stage all
```

The primary config evaluates zeroth-order, serial first-order (Mode 3), and
batched first-order (Mode 4). Stage B records clean/dirty counts, reference
age, RMS/mean/max force errors, P95/P99 per-atom errors, energy errors,
message errors, finiteness, timing, and memory.

To resume only completed stages:

```bash
python experiments/run_campaign.py \
  --config configs/generalization.json --stage all --resume
```

The runner refuses partial non-empty output directories and mismatched resume
configurations.

### P1. Mode 3/Mode 4 and chunk validation

Use the same Stage-A trajectory, snapshots, tolerances, and output roots for
both modes:

```bash
python -m frost.e1.stage_b_accuracy_sampling \
  --input results/active/pilot/diamond_si_T300_seed42/stage_a/per_timestep.csv \
  --trajectory results/active/pilot/diamond_si_T300_seed42/stage_a/trajectory.traj \
  --sample-count 5 --tolerances 0.005,0.01 \
  --cache-mode 3 --output-dir results/active/pilot/stage_b_mode3

for chunk in 64 128 256 512 1024; do
  python -m frost.e1.stage_b_accuracy_sampling \
    --input results/active/pilot/diamond_si_T300_seed42/stage_a/per_timestep.csv \
    --trajectory results/active/pilot/diamond_si_T300_seed42/stage_a/trajectory.traj \
    --sample-count 5 --tolerances 0.005,0.01 \
    --cache-mode 4 --batched-action-chunk-size "$chunk" \
    --output-dir "results/active/pilot/stage_b_mode4_chunk${chunk}"
done
```

Mode 2 is a dense-Jacobian validation oracle only:

```bash
python -m frost.e1.stage_b_accuracy_sampling \
  --input results/active/pilot/diamond_si_T300_seed42/stage_a/per_timestep.csv \
  --trajectory results/active/pilot/diamond_si_T300_seed42/stage_a/trajectory.traj \
  --sample-count 5 --tolerances 0.005,0.01 --cache-mode 2 \
  --jacobian-chunk-size 64 \
  --output-dir results/active/pilot/stage_b_dense_oracle
```

### P1. Frontier statistics and edge-level analysis

Block bootstrap one trajectory at a time:

```bash
python -m frost.e1.frontier_analysis \
  --input results/active/generalization/diamond_si_T300_seed42/stage_b_mode4/accuracy_results.csv \
  --block-length 10 --draws 5000 \
  --output-dir results/active/frontier_seed42_mode4
```

W0 now produces an instrumented per-edge `message_errors.csv` containing
`edge_displacement_A`, `reference_age_steps`, and `message_error`. Aggregate
Stage-B CSVs are rejected rather than expanded into fabricated edge rows:

```bash
python -m frost.e1.edge_error_analysis \
  --input results/active/w0_smoke/message_errors.csv \
  --output-dir results/active/w0_smoke/edge_error
```

### P1. E4/W4 fixed-configuration performance

Run only after P0 correctness and core accuracy:

```bash
python -m frost.e1.inference_benchmark \
  --trajectory results/active/generalization/diamond_si_T300_seed42/stage_a/trajectory.traj \
  --steps 50 250 500 750 950 --n-frames 5 \
  --warmup 10 --evaluations 100 --repeats 3 \
  --cache-mode 4 --batched-action-chunk-size 256 \
  --output-dir results/active/w4_inference_seed42
```

This includes exact, disabled, all-dirty, controlled 0/25/50/75/100%,
natural threshold, serial first-order, and batched first-order arms. It
records synchronized full-calculator timing, diagnostic component timing,
peak allocated/reserved CUDA memory, and `cost_components.csv`.

Analyze the measured cost model:

```bash
python -m frost.e1.cost_model_analysis \
  --input results/active/w4_inference_seed42/cost_components.csv \
  --output results/active/w4_inference_seed42/cost_model.csv
```

Full-calculator time is the primary result. Component times must not be added
when they overlap.

### P1. W4.1 integrated inference hot-path optimization

W4.1 preserves the historical W4 output and writes to a new directory. Read
the execution-path audit before measuring:

```bash
cat docs/W4_1_IMPLEMENTATION_AUDIT.md
python -m frost.e1.inference_benchmark \
  --trajectory results/active/generalization/diamond_si_T300_seed42/stage_a/trajectory.traj \
  --steps 50 250 500 750 950 --n-frames 5 \
  --warmup 10 --evaluations 100 --repeats 3 \
  --cache-mode 4 --batch-size 1024 \
  --output-dir results/active/w4_1_inference_seed42
python -m frost.e1.cost_model_analysis \
  --input results/active/w4_1_inference_seed42/cost_components.csv \
  --output results/active/w4_1_inference_seed42/cost_model.csv
```

The W4.1 summary records actual eligible, dirty, substituted, and
first-order-action edge counts plus runtime environment metadata. Compare
only synchronized complete-calculator latency with the historical W4 result;
component timings are diagnostic and may overlap. See
`docs/W4_1_RESULTS.md` for the reporting requirements.

### P2. E5/W5 matched-start MD rollout

Run a short rollout before a long rollout:

```bash
python -m frost.e1.paper_benchmark \
  --model mace-mp-0 --device cuda --dtype float64 \
  --structure diamond_si --temperature 300 \
  --seeds 42 --warmup-steps 10 --production-steps 25 \
  --cache-mode 4 --batched-action-chunk-size 256 \
  --tolerances 0.005 0.01 \
  --output-dir results/active/w5_rollout_smoke
```

Then run the required matched-start independent-seed rollout:

```bash
python -m frost.e1.paper_benchmark \
  --model mace-mp-0 --device cuda --dtype float64 \
  --structure diamond_si --temperature 300 \
  --seeds 42 137 271 --warmup-steps 100 --production-steps 500 \
  --cache-mode 4 --batched-action-chunk-size 256 \
  --tolerances 0.005 0.01 0.02 \
  --output-dir results/active/w5_rollouts
```

The output includes matched initial-state `.npz` artifacts and hashes,
per-step/cumulative time, temperature, kinetic/potential/total energy, energy
drift, reuse, reference age, exact-force checks, and finiteness. Do not claim
long-time trajectory identity; compare distributions and same-configuration
force accuracy.

### P2. Generalization

Temperature (100/300/600 K, independent seeds):

```bash
python experiments/run_campaign.py \
  --config configs/temperature.json --stage all
```

System-size scaling (64/216/512 atoms):

```bash
python experiments/run_campaign.py \
  --config configs/scale.json --stage all
```

### P3. Evidence summaries and figures

These commands only read completed raw results. Missing experiments remain
missing and cause the corresponding summarizer to fail rather than fabricate
rows:

```bash
python -m analysis.summarize_stage_a
python -m analysis.summarize_frontier
python -m analysis.summarize_generalization
python -m analysis.summarize_inference_benchmark
python -m analysis.summarize_rollout
python -m analysis.generate_paper_figures
```

Outputs belong under:

```text
results/active/paper_evidence/
  stage_a_summary.csv
  frontier_summary.csv
  generalization_summary.json
  inference_cost_summary.json
  rollout_summary.json
  figures/
  tables/
```

## Scientific constraints

- W0 is an acceptance gate, not a descriptive afterthought.
- Stage A clean fraction is a geometric upper bound, not force accuracy.
- All-clean and controlled clean-fraction arms are diagnostic controls.
- Exact-vs-cache error is not model-vs-DFT error.
- Independent trajectories are replication units; snapshots are correlated.
- Never overwrite `results/legacy_baseline/` or a successful active run.
- Do not claim measured speedups, break-even points, or generalization until
  the corresponding GPU outputs exist and pass validation.
