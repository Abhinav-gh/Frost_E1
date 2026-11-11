# W4.1 Results

## Status

The W4.1 code and instrumentation changes are implemented, but no GPU
benchmark was run by the assistant. Historical W4 results were not modified.
Run the command below on the target GPU to produce the W4.1 artifact set.

## Changes

- Added substitution-point edge accounting for eligible, dirty, substituted,
  first-order, exact-message, and cached-message edges.
- Avoided copying exact and substituted message tensors for normal timed runs;
  full message snapshots remain available through explicit audit capture.
- Added trajectory, step, PyTorch, CUDA, and GPU metadata to W4 summaries.
- Added this execution-path audit and documented the remaining duplicate-forward
  and geometry-reconstruction bottlenecks.

## Required measurement

```bash
python -m frost.e1.inference_benchmark \
  --trajectory results/active/generalization/diamond_si_T300_seed42/stage_a/trajectory.traj \
  --steps 50 250 500 750 950 --n-frames 5 \
  --warmup 10 --evaluations 100 --repeats 3 \
  --cache-mode 4 --batch-size 1024 \
  --output-dir results/active/w4_1_inference_seed42
```

Then analyze the generated cost components:

```bash
python -m frost.e1.cost_model_analysis \
  --input results/active/w4_1_inference_seed42/cost_components.csv \
  --output results/active/w4_1_inference_seed42/cost_model.csv
```

Report absolute and percentage latency changes only from synchronized
full-calculator timings. Do not infer acceleration from component sums or
zero-valued targeted-savings fields.
