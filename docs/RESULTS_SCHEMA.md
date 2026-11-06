# Results schema

## Stage A

`per_timestep.csv` contains `step`, `tolerance_A`, `n_edges`,
`n_clean`, `n_dirty`, `n_new`, `n_removed`, `n_refreshed`, `phi`,
`clean_fraction`, displacement percentiles, and `generation_max`.
`summary.json` records configuration, environment, initial-state hash, and
the explicit statement that Stage A has no force-error measurement.

## Stage B

`accuracy_results.csv` contains one row per selected
`(step, tolerance_A)` pair: edge counts, energy error, RMS/mean/max and
per-atom P95/P99 force error, clean/dirty message error, reference age,
relative force error, finiteness, and timing. `summary.json` repeats the
configuration and grouped descriptive statistics.

## W0, W4, and W5

- W0: `audit_summary.json`, `audit_checks.csv`, `message_errors.csv`,
  `force_errors.csv`, `audit.log`, and `metadata.json`; failed controls return
  nonzero and are never represented as blanks.
- W4/E4: `timings.csv`, `component_timings.csv`, `cost_components.csv`,
  `break_even_summary.json`, and `summary.json`. Complete synchronized
  calculator time is primary; component rows are diagnostic.
- W5/E5: `frame_metrics.csv`, `cache_metrics.csv`, `manifest.json`, and one
  `initial_state_seed<N>.npz` per seed. Frame rows include cumulative time,
  thermal/total energy, drift, reuse, reference age, and exact-force checks.

## Per-edge analysis input

[`edge_error_analysis.py`](../frost/e1/edge_error_analysis.py) requires a
separate instrumented CSV with `edge_displacement_A`,
`reference_age_steps`, and `message_error`. Optional columns such as
`edge_id`, `radial_distance_A`, `orientation_change`, `cache_mode`, and
`clean` are preserved and can be grouped downstream. Aggregate Stage-B rows
are rejected rather than expanded into fake edge data.
