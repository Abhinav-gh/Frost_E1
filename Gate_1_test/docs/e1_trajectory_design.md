# E1 Trajectory Cache Design

## 1. Current implementation architecture

The repository has already validated the core edge-local interception mechanism. The fix is not implemented by moving atoms or creating a mixed-time structure. Instead, the code now operates on MACE's exact first-layer `mji` message before the scatter aggregation step.

The core components are:

- `frost/e1/mace_patch.py`: temporary monkey-patch of the first unfused MACE interaction block
- `frost/e1/force_error.py`: exact-force evaluation and edge-local cached-force evaluation
- `frost/e1/force_error_tests.py`: seven integrity tests for the edge-local correctness harness
- `frost/e1/reference_cache.py`: per-edge geometric reference bookkeeping
- `frost/e1/__main__.py`: trajectory driver hook for MD + sampled exact-vs-cached comparisons

## 2. Cache state representation

The current design keeps a per-edge cache entry with:

- edge identity (`src`, `dst`, periodic shift)
- reference displacement vector `r_ref`
- reference scalar distance `d_ref`
- reference direction `rhat_ref`
- reference timestep
- age and generation counters
- validity flag

This is the stage-1 version of the Frost-style cache state and is intentionally simpler than the final full runtime.

## 3. Edge identity

The edge key is a canonical physical identity, not a neighbor-list array index:

- `(i, j, shift)`

The shift is necessary because the same pair of atoms may connect via different periodic images. This matters for PBC correctness and for matching the actual MACE neighbor list.

For the current MACE version, `edge_index[0]` is the sender atom and
`edge_index[1]` is the receiver atom. MACE forms the directed vector as
`positions[receiver] - positions[sender] + shifts`, where `unit_shifts` is the
integer fractional cell offset and `shifts` is its Cartesian form. The mapping
utility in `frost/e1/geometry.py` canonicalizes both ASE and MACE as
`(sender, receiver, unit_shift)`.

On the fixed diamond-Si configuration used for low-level validation, ASE and
MACE each produced 2,944 directed edges. Their arrays were ordered differently,
but the canonical key sets had no duplicates, no missing keys, and matched
exactly.

## 4. Refresh policy

The current trajectory cache is still a geometric E1 simulation layer rather than a finalized production Frost runtime. It evaluates clean vs dirty edges using the existing tolerance predicate and refreshes dirty edges per pass. The semantics are intentionally conservative and match the current E1 harness level.

The prototype computes all tolerance dirty masks from one pre-step reference
snapshot before applying any refresh. This prevents a later tolerance from
depending on an earlier tolerance's refresh. Reference geometry is still
shared by this prototype; fully independent per-tolerance cached message values
remain future work.

## 5. Handling new and removed edges

- New edge: treated as dirty and inserted into the cache.
- Removed edge: invalidated and removed from active cache state.
- Existing edge: evaluated against its own reference state.

This is consistent with the E1 requirement that the cache be edge-local and dynamic.

## 6. Exact-vs-cached force semantics

For sampled frames, the intended force comparison is:

- exact: `F_exact = MACE(R_t)`
- cached: `F_cached = MACE_with_edge_cache(R_t)`

The same coordinates, same cell, and same PBC must be used for both. The only difference is the selective replacement of layer-1 `mji` values on clean edges.

## 7. Zeroth-order and first-order modes

The implementation supports both:

- zeroth order: `g_hat = g_ref`
- first order: `g_hat = g_ref + J_ref @ delta_r`

The first-order mode is the scientifically relevant Frost candidate. Zeroth-order is retained as a negative control and sanity check.

## 8. Remaining limitations

The current trajectory hook still does not yet constitute a fully production-quality per-edge refresh engine for all eight tolerances. Specifically:

- the edge cache is not yet fully independent per epsilon in every code path
- the sampled frame callback is still a staged E1 harness rather than the final 10,000-step controller
- full 20 ps + 10,000-step E1 production logging is still pending

## 9. Status

The underlying mechanism is now valid at the edge-interception level and is validated by the seven-case integrity matrix. The remaining work is trajectory integration and full E1 production execution, not another redesign of the mathematical repair.

The staged low-level validation also includes explicit NEW/VALID/DIRTY/REMOVED
lifecycle reporting, cache hit/miss counts, a no-Jacobian layer-1 message
capture path, ASE calculator-cache invalidation before intercepted evaluations,
and cache tensor device/dtype normalization for CUDA MACE execution. The mode-0
exact-control path passed on one fixed configuration and a three-step Si canary.
