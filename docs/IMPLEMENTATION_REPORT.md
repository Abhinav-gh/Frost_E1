# Implementation report

## A. Architecture

The repository now separates geometry/cache/message substitution from model
loading, experiment runners, and post-run analysis. Stage A produces a
trajectory and geometry-only reuse statistics. W0 audits the integrated
substitution path. Stage B evaluates selected snapshots against exact MACE.
W4 measures fixed-configuration complete-calculator cost, and W5 measures
matched-start rollout behavior.

## B. Implemented features

- Added the required implementation audit and architecture/protocol/schema
  documentation.
- Added Stage-A `metadata.json`, `reuse_summary.csv`, and `run.log` outputs.
- Added explicit edge-level analysis with schema validation and correlation
  summaries, backed by per-edge W0 message-error output.
- Added a measured-versus-predicted cost-model analysis utility.
- Added campaign configuration validation, campaign manifests, structured
  command logging, completion markers, and safe resume behavior.
- Added smoke, W4/E4, and W5/E5 configuration files and explicit execution
  policy fields.
- Added evidence aggregation modules and a raw-results-only figure generator.
- Added CPU-only tests for the new analysis utilities.
- Updated the README with setup, all experiment commands, resume rules, and
  evidence interpretation.

## C. Optimizations

Mode 4 chunked batched JVP/VJP support and canonical edge-key matching were
already present and are documented as available validation paths. No new
runtime optimization is claimed in this phase because GPU benchmarks were not
run. The optimization log records this limitation explicitly.

## D. Scientific correctness

The existing W0 implementation covers exact/disabled/all-dirty behavior,
threshold zeroth- and first-order paths, batched first-order behavior,
controlled perturbations, message capture, storage separation, and
zero-displacement controls. Campaign `all` and explicit `stage-b` runs now
require a passing W0 summary before Stage B.

No numerical or GPU result is asserted by this report.

## E. Testing

The new tests cover negative cost components, cost residual calculation, and
rejection of aggregate Stage-B rows by the edge analyzer. GPU experiments were
not run. In the available environment, `python` resolved to an older
interpreter and `pytest` was unavailable; use the repository VM's Python 3.10+
environment. The exact CPU commands are:

```bash
python3 -m compileall -q frost experiments analysis
python3 -m pytest -q
```

## F. Experiment infrastructure

Stage A, Stage B, W0, W4, and W5 remain runnable through their existing module
CLIs. The campaign runner supports `all`, `stage-a`, `audit`, and `stage-b`,
with validated JSON configuration, fresh-output protection, W0 gating, and
`--resume`.

## G. Research readiness

Ready for GPU execution after environment validation:

- pilot Stage A -> W0 -> Stage B;
- independent-seed generalization;
- temperature and scale configurations;
- mode 3 versus mode 4 comparison;
- fixed-configuration W4 timing;
- matched-start W5 rollout.

Still requiring GPU/instrumentation work:

- measured mode-3/mode-4 speed and memory comparison;
- full per-edge message-error export for edge-level analysis;
- cost-model input populated from measured component timings;
- paper figure/table aggregation across completed independent campaigns.

Existing legacy results remain untouched. New output must use unique directories
under `results/active/`.

## H. Exact next commands

Run these on the GPU VM, from the repository root, after installing a
CUDA-compatible PyTorch and MACE build:

```bash
python -m pytest -q
python experiments/run_campaign.py --config configs/pilot.json --stage all
python experiments/run_campaign.py --config configs/generalization.json --stage all
python experiments/run_campaign.py --config configs/scale.json --stage all
python experiments/run_campaign.py --config configs/temperature.json --stage all
python experiments/run_campaign.py --config configs/generalization.json --stage all --resume
python -m frost.e1.inference_benchmark \
  --trajectory results/active/generalization/diamond_si_T300_seed42/stage_a/trajectory.traj \
  --steps 50 250 500 750 950 --n-frames 5 --warmup 10 --evaluations 100 --repeats 3 \
  --output-dir results/active/w4_inference_seed42
python -m frost.e1.paper_benchmark \
  --structure diamond_si --temperature 300 --seeds 42 137 271 \
  --warmup-steps 100 --production-steps 500 \
  --tolerances 0.005 0.01 0.02 \
  --output-dir results/active/w5_rollouts
```

Inspect every W0 `summary.json` before interpreting its corresponding Stage B
results. Generate edge-error and cost-model summaries only from their declared
raw schemas.
