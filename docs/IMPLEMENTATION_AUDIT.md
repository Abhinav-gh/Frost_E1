# Implementation audit

## Scope and baseline

This audit records the implementation that was present before this upgrade.
No GPU experiment was run while preparing it. Existing files under
`results/legacy_baseline/` are historical artifacts and are not modified.

| Area | Current implementation | Status |
|---|---|---|
| Geometry and periodic edges | `frost/e1/geometry.py` and ASE neighbor-list keys | Implemented and unit-tested |
| Per-edge reference lifecycle | `PerEdgeReferenceCache` | Implemented; refreshes dirty/new edges |
| MACE edge ordering | Canonical `(sender, receiver, unit_shift)` key comparison | Implemented; mismatches fail loudly |
| Exact/Frost substitution | `force_error.py`, `mace_patch.py`, `IntegratedFrostCalculator` | Implemented |
| Zeroth-order and serial first-order | Cache modes 1 and 3 | Implemented |
| Batched first-order | Cache modes 4, chunked JVP/VJP | Implemented; GPU validation remains to be run |
| Dense Jacobian reference | Stage-B mode 2 | Implemented as a comparison path |
| W0 audit | `frost/e1/e3_audit.py` | Implemented; must gate Stage B |
| Stage A | Geometry-only trajectory replay and CSV | Implemented; no Jacobians |
| Stage B | Stratified sampling and exact-vs-cache metrics | Implemented |
| W4 benchmark | Fixed-frame arms and synchronized full-evaluation timing | Implemented; component slices are diagnostic |
| W5 rollout | Matched post-equilibration state and force checks | Implemented |
| Statistical summary | Block bootstrap frontier summary | Implemented |
| Edge-level analysis | New `edge_error_analysis.py` | Requires an instrumented per-edge CSV |
| Cost model | New `cost_model_analysis.py` | Requires measured component CSV |
| Campaign orchestration | Existing subprocess runner | Hardened in this upgrade |

## Highest-priority risks

1. MACE and PyTorch versions on the GPU VM must be validated before interpreting
   numerical or performance results.
2. Stage-B output contains aggregate clean-edge message errors, not one row per
   edge. Edge-level conclusions require an instrumented run with the schema
   documented in `docs/RESULTS_SCHEMA.md`.
3. Full-calculator wall time, not an isolated message-layer time, is the
   performance metric. CUDA synchronization is required around timed regions.
4. The all-clean arm is diagnostic only and must not be used as a threshold
   policy.
5. Existing worktree edits to the generalization configuration and benchmark
   defaults predate this change and are preserved.

## File-by-file implementation plan

1. Keep geometry/cache/message substitution as separate core layers.
2. Use W0 as an explicit correctness gate before Stage B.
3. Keep Stage A derivative-free and save a trajectory for Stage B.
4. Keep Stage B output schemas stable and summarize with block bootstrap.
5. Use separate analysis modules for edge errors and measured cost modeling.
6. Make campaign runs resumable through completion markers and reject
   incomplete or conflicting outputs.
7. Document exact GPU commands without executing them here.

## Validation performed

The CPU unit suite is the only execution validation performed for this
upgrade. GPU model loading, CUDA timing, MD, W0, Stage A, Stage B, W4, and W5
experiments are intentionally left for the user's GPU VM.
