# Frost edge caching: experiment workspace

This repository studies when per-edge message reuse in MACE is accurate and whether the full cache path reduces end-to-end cost. The initial evidence is negative on integrated runtime and mixed on reuse versus force error. The experiments are designed to characterize those limits; a speedup is not assumed.

## Repository map

| Path | Contents |
|---|---|
| `frost/e1/` | Reusable cache, geometry, MACE patch, trajectory, audit, and benchmark code |
| `experiments/` | Campaign runner for repeated Stage A/B studies |
| `configs/*.json` | Pilot, seed, temperature, tolerance, system-size and model matrices |
| `docs/EXPERIMENT_ROADMAP.md` | Ordered study plan, evidence gates, and claim limits |
| `docs/reference/` | Prior project reference PDFs |
| `docs/archive/` | Prior plans and notes, kept for provenance |
| `results/legacy_baseline/` | Existing result artifacts, preserved unchanged |
| `results/active/` | New run outputs; ignored by Git by default |
| `tests/` | Existing unit tests (not run as part of this edit) |

## VM setup

Use Python 3.10+ on the GPU VM. Install a PyTorch build compatible with that VM's CUDA driver, then install MACE and this package. The correct PyTorch wheel depends on the VM; do not copy the workstation's CUDA wheel blindly.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the CUDA-compatible PyTorch build for this VM, then:
python -m pip install mace-torch
python -m pip install -e .
```

Check that `python -c "import torch; print(torch.cuda.is_available())"` reports `True` before a production run. MACE-MP-0 may download its checkpoint on first use; arrange network access or the corresponding cache on the VM.

## Recommended execution order

Run from the repository root. Give each launch a fresh output directory; tools refuse to overwrite non-empty directories. The campaign's `all` stage executes Stage A, runs W0 audits for each case, stops if the audit fails, and only then starts Stage B.

### 1. Reproduce independent 300 K trajectories and sparse accuracy frontiers

First run the small pipeline pilot on the GPU VM. It exercises Stage A, the W0 gate, and both Stage-B modes with five snapshots:

```bash
python experiments/run_campaign.py --config configs/pilot.json --stage all
```

If that completes and the generated `results/active/pilot/.../w0_audit/summary.json` reports `"audit_passed": true`, proceed to the larger campaigns below.

The checked-in campaign config runs seeds 42, 43, and 44 for 1000 production steps, then evaluates 50 time-stratified snapshots at seven tolerances using zeroth- and first-order reuse.

```bash
python experiments/run_campaign.py --config configs/generalization.json --stage all
```

After the primary 64-atom study, use the separate scale config for the 64/216/512-atom size comparison:

```bash
python experiments/run_campaign.py --config configs/scale.json --stage all
```

The 100/300/600 K extension with three seeds per temperature is in `configs/temperature.json`:

```bash
python experiments/run_campaign.py --config configs/temperature.json --stage all
```

For a smaller smoke-sized GPU run, copy the JSON config to a new file and reduce `seeds`, `production_steps`, `accuracy_sample_count`, and `tolerances_A`; do not label that output as paper evidence. The campaign writes one Stage A trajectory and CSV per system/temperature/seed, then separate Stage B outputs per cache mode. To resume Stage B after Stage A has completed:

```bash
python experiments/run_campaign.py --config configs/generalization.json --stage stage-b
```

### 2. Audit the message substitution path before interpreting cache accuracy

Use a trajectory from Stage A. The audit compares exact and actually substituted per-edge messages on clean and dirty edges, checks separate captured storage, and perturbs one atom by controlled displacements. It includes exact, disabled, threshold zeroth-order, threshold first-order, and intentionally all-clean diagnostic arms.

```bash
python -m frost.e1.e3_audit \
  --trajectory results/active/generalization/diamond_si_T300_seed42/stage_a/trajectory.traj \
  --step 100 --tolerances 0.005 0.01 --output-dir results/active/w0_audit_seed42
```

Treat the deliberately all-clean arm as a failure-revealing diagnostic, not a valid cache policy. Zeroth-order message substitution preserves message values but drops the position derivative through reused messages, so its zero-displacement force error is expected and reported separately. The W0 force gate applies to the exact path, disabled-cache path, and first-order path. Review `audit.csv`, `summary.json`, dirty-edge exactness, and finite outputs before interpreting the frontier.

### 3. Summarize accuracy and uncertainty

Stage B emits per-snapshot exact-versus-Frost energy and force errors. Summarize temporal dependence with a block bootstrap:

```bash
python -m frost.e1.frontier_analysis \
  --input results/active/generalization/diamond_si_T300_seed42/stage_b_mode3/accuracy_results.csv \
  --block-length 10 --draws 5000 --output-dir results/active/frontier_seed42_mode3
```

Bootstrap intervals over one trajectory do not replace independent seeds. Report per-trajectory results and use the independent trajectories as the replication unit in the paper.

### 4. Measure full-calculator cost at fixed configurations

This benchmark uses saved frames, at least 10 warmups, at least 100 timed evaluations, 3 repeats, and CUDA synchronization around every complete calculator evaluation. It contrasts exact MACE, disabled reuse, dirty-only work, 0/25/50/75/100% clean-message reuse, and first-order cache paths. It records host-side cache component timing slices and peak allocated CUDA memory; use synchronized full-calculator wall time for performance comparisons.

```bash
python -m frost.e1.inference_benchmark \
  --trajectory results/active/generalization/diamond_si_T300_seed42/stage_a/trajectory.traj \
  --steps 50 250 500 750 950 --n-frames 5 --warmup 10 --evaluations 100 --repeats 3 \
  --output-dir results/active/w4_inference_seed42
```

This isolates repeated inference at fixed configurations; it is not an MD rollout timing. Report `timings.csv` distributions and compare with complete MD timings from `frost.e1.end_to_end_benchmark` only when both methods use the same saved starting state and measured scope.

For a batched Stage-B derivative alternative, run the same pilot snapshots with `--cache-mode 2`. This uses chunked `vmap(jacrev)` Jacobians and tensor contractions instead of per-edge JVP/VJP closures. Compare accuracy with mode 3, compare `total_stage_b_runtime_seconds` and `peak_cuda_allocated_bytes`, then sweep chunk sizes (64, 128, 256) in separate output directories. Keep the chunk size that improves runtime without changing the error results materially; the 49 GB card provides headroom for this experiment.

### 5. Compare longer exact and cached MD rollouts

The paper benchmark equilibrates once with exact MACE and copies the same positions and momenta into each production arm. It records synchronized complete-step wall time, exact force checks on produced frames, cache metrics, and the initial-state hash.

```bash
python -m frost.e1.paper_benchmark \
  --structure diamond_si --temperature 300 --seeds 42 137 271 \
  --warmup-steps 100 --production-steps 500 \
  --tolerances 0.005 0.01 0.02 \
  --output-dir results/active/w5_rollouts
```

This gives three stochastic realizations at the selected temperature. For any higher-temperature or second-material rollout, first confirm exact-MACE stability and use a separate output directory.

## Scope and interpretation

- Stage A measures geometry reuse only. Its clean fraction is not a force-accuracy or runtime result.
- Stage B evaluates sampled frames against exact MACE. Use temporal blocks and independent trajectories when reporting uncertainty.
- Current trained/model coverage is centered on MACE-MP-0 and the available ASE systems. Generalization across materials requires validating that the selected pretrained model is stable and chemically appropriate first.
- The edge-cache path includes Python, geometry, cache management, and derivative-action overhead. Report whole-calculator and whole-trajectory timings, not only the replaced layer operation.
- Keep model/checkpoint identity, commit, Python/PyTorch/CUDA/GPU, seed, structure, temperature, dtype, tolerance, and initial-state provenance with every result.
- Do not claim universal slowdown, universal cache failure, or generalization from the pilot. The intended paper is a careful characterization of where geometric reuse does and does not translate into accurate, useful acceleration.

## Existing results

Previously generated files are under `results/legacy_baseline/`. They remain available for comparison, but new runs go to `results/active/` and are not committed by default. Never merge results from different timing scopes into a single speedup statistic.
