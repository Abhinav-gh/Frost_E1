# Frost negative-results paper: plan from the current codebase

## Working paper question

Under what conditions does temporal reuse of first-layer, edge-local geometric messages in MACE produce a useful accuracy–runtime tradeoff for molecular-dynamics workloads? The paper should report a negative result only where the experiment distinguishes a failed caching hypothesis from an incomplete or incorrect implementation.

## Current status and claim boundary

The repository has a real MACE first-layer interception path and a per-edge geometry cache. Existing Stage-B snapshots show a reuse/force-error tradeoff on 64-atom diamond silicon, while the short integrated run shows severe overhead. These are pilot observations: accuracy frames are sparse and the short run is too brief for stable timing. The new paper runner uses one exact-MACE equilibration per seed, then starts each production arm from identical saved positions and momenta. The paper should not claim a general failure of edge caching yet.

The historical atom-freezing experiment and the pathological elementwise dense-Jacobian loop are methodology diagnostics. Do not use them as scientific evidence about edge-local caching.

## Work package A — correctness and protocol repair

1. The new runner performs one exact-MACE equilibration per seed, then reuses that exact post-equilibration state for exact and cached production arms; the state is hashed into the manifest.
2. Confirm runtime edge-key alignment (checked on every cached call), neighbor-list changes, reference refresh counts, and cache generations in `cache_metrics.csv`.
3. Resolve and document why the short end-to-end report shows zero force error while sparse Stage-B snapshots show nonzero error. Compare identical snapshots, tolerance, cache mode, refresh semantics, and reference state.
4. Add unit/integrity checks for edge-key alignment, message replacement, zero-displacement equivalence, exact fallback for new/dirty edges, and finite forces.

## Work package B — accuracy/reuse frontier

Run `frost.e1.paper_benchmark` on the GPU VM with MACE-MP-0, first on 64-atom diamond Si at 300 K. Use multiple independent seeds and a production segment long enough to sample varied displacements. Sweep 0.001, 0.002, 0.005, 0.01, 0.02, and 0.05 Å. Evaluate every production frame against exact MACE and report force RMS plus atomwise p50/p95/p99/max, energy error, clean-edge fraction, edge churn, and finite/stability status. Run both cache modes (zeroth and first order).

Then repeat the most informative thresholds at a second temperature and on at least one chemically distinct benchmark. Only generalize beyond MACE/Si if these replications support it.

## Work package C — end-to-end cost and dynamics

Measure exact and cached throughput after warm-up, with full wall time including geometry, cache actions, refreshes, synchronization, and bookkeeping. Repeat runs and report median and dispersion. Profile cache construction and first-order JVP/VJP costs separately, plus peak memory. Run paired exact/cached trajectories from identical initial states for longer horizons; compare temperature, potential-energy drift, structural observables, and failure rate.

## Work package D — paper and reproducibility

Archive environment, model/checkpoint identity, hardware, commit, configs, seeds, raw frame metrics, cache metrics, and scripts. Analyze uncertainty across independent seeds (not just frames from one trajectory). Include a cost-versus-error frontier, error distributions, reuse distributions, and a runtime breakdown. State sample sizes and limitations.

## Decision gates

- **Correctness gate:** resolved force mismatch and integrity checks pass before interpreting any performance result.
- **Utility gate:** at least one operating point must meet a predeclared force-error tolerance while reducing end-to-end time; otherwise report the measured failure and explain whether it comes from low safe reuse, derivative-action cost, or bookkeeping.
- **Scope gate:** if only one material/model was run, title and conclusions must remain specific to that setting.

## First VM run

From `Gate_1_test`, use the project's GPU environment and run a short pilot first:

```bash
python -m frost.e1.paper_benchmark --seeds 42 --warmup-steps 10 --production-steps 20 --tolerances 0.005 0.01 --output-dir results/paper_pilot
```

Review `manifest.json`, `frame_metrics.csv`, and `cache_metrics.csv`; then run the full baseline:

```bash
python -m frost.e1.paper_benchmark --seeds 42 137 271 --warmup-steps 100 --production-steps 500 --tolerances 0.001 0.002 0.005 0.01 0.02 0.05 --output-dir results/paper_si_300K
```

Repeat with `--cache-mode 1` for zeroth-order reuse. Use separate output folders for every model/system/temperature/configuration. Do not overwrite prior results.
