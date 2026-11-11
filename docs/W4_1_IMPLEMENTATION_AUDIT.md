# W4.1 Integrated Inference Implementation Audit

## Scope

This audit describes the checkout used for W4.1. It is an execution-path
audit, not a performance claim. The historical W4 results remain unchanged.
No GPU benchmark was run while making this change.

## Current call graph

`inference_benchmark.run()` creates an `IntegratedFrostCalculator` for each
fixed frame and arm, then calls `Atoms.get_forces()` repeatedly.

1. `IntegratedFrostCalculator.calculate()` copies the `Atoms` object.
2. `build_edge_geometry_from_ase()` rebuilds an ASE neighbor list and creates a
   Python dictionary keyed by `(sender, receiver, periodic_shift)`.
3. `extract_layer1_local_inputs()` constructs a MACE batch and executes one
   model forward with the first interaction patched only to capture local
   inputs. It captures edge vectors, node features/attributes, edge indices,
   unit shifts, and exact first-layer messages.
4. The calculator maps the captured MACE edge keys to the ASE edge-key set and
   walks the cache entries to classify, refresh, and assemble references.
5. `compute_cached_forces()` clears the ASE calculator result, patches the first
   interaction again, and executes a second model forward. Clean messages are
   substituted before first-layer scatter aggregation; dirty messages use the
   current exact first-layer result. MACE then performs the remaining
   interactions, energy calculation, and force autograd.

Therefore the current Frost path has two model forwards per evaluation. The
first is not redundant for correctness in the current architecture because it
provides the tensors needed to classify and prepare the cache before the
second forward. Eliminating it requires moving cache preparation into the
actual forward and committing refreshed references after the first-layer
message is captured. That architectural change is not silently claimed here.

## Hot-path inventory

- **Geometry and neighbor lists:** ASE's `primitive_neighbor_list` is rebuilt
  for every calculator evaluation. MACE separately constructs its own batch
  and neighbor tensors.
- **Edge identity:** MACE edge tensors are converted to Python canonical keys;
  the key set is compared with the ASE geometry keys. A Python key-to-index
  dictionary is then used for cache lookup.
- **Cache operations:** `PerEdgeReferenceCache.step()` uses Python dictionaries
  and per-edge loops for perturbations, dirty classification, refresh, age
  updates, and removal. The integrated calculator performs another per-edge
  loop to copy reference tensors and actions into MACE-order tensors.
- **Tensor allocations:** Each evaluation allocates current/reference message
  and vector tensors. Mode 4 also allocates source-feature and edge-index
  reference tensors, then gathers clean indices and concatenates chunk outputs.
- **Mode 4 actions:** clean edges only are passed to chunked matrix-free JVP/VJP
  operations. No dense Jacobian is materialized.
- **Synchronization:** W4 synchronizes immediately before and after each timed
  full-calculator evaluation. The normal calculator path does not add a CUDA
  synchronization for each edge.
- **Substitution point:** replacement happens in the patched first interaction,
  after `conv_tp` computes exact `mji` and before `scatter_sum` aggregates it.

## W4.1 instrumentation

Timed calculator records now expose eligible clean, dirty, substituted,
first-order-action, exact-message, and cached-message edge counts. These
counts are produced at the substitution point and do not require copying all
message tensors when audit capture is disabled. `summary.json` also records
trajectory, steps, PyTorch/CUDA versions, CUDA availability, and GPU name.

The complete synchronized calculator latency remains authoritative. Component
timings are diagnostic and must not be added unless their inclusivity is
explicitly established.

## Remaining optimization work

The highest-impact unimplemented optimization is forward integration:
prepare cache metadata from the batch, capture first-layer inputs during the
single real model forward, and refresh only after the captured exact messages
are available. Geometry reuse and persistent GPU buffers likewise require
explicit topology and periodic-shift invalidation tests before adoption.
