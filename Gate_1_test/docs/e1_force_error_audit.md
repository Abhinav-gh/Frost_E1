# E1 Force-Error Methodology Audit

## A. What the current implementation actually does

Currently, the E1 measurement proxies force error by intercepting the system *before* it enters MACE. 
In `frost/e1/force_error.py`:
1. It compares the current exact edge vectors $r_{ij}$ against reference edge vectors $r_{ij}^{ref}$.
2. It tests if the perturbation exceeds the tolerance $\epsilon$. If so, the edge is "dirty".
3. It determines an atom to be "stale" if **all** of its incident edges are clean (do not exceed $\epsilon$).
4. For all atoms flagged as stale, it replaces their current coordinates $r_i$ with their old reference coordinates $r_i^{ref}$.
5. It feeds this artificially mixed-time configuration (some atoms current, some atoms in the past) into `MACECalculator.get_forces()`.

**Why this is not equivalent to an edge-local cache:**
When you hold an entire atom at $r_i^{ref}$ while a neighboring atom $j$ moves to $r_j(t)$ (because the edge between them is clean enough not to flag $j$ as stale), you are physically stretching/straining the lattice. MACE is rotationally and translationally invariant, but it evaluates the total structural energy. The mixed-time configuration represents a high-energy non-physical transition state that MACE violently penalizes with massive force errors. 

Frost, conversely, evaluates the entire model on the perfectly pristine current configuration $R(t)$, but selectively replaces the *abstract intermediate geometric representations* for edges that didn't change much relative to each other. Frost never physically moves atoms.

## B. Where the actual MACE layer-1 cacheable computation lives

MACE evaluates its zero-closure (cacheable) edge geometry inside its first Interaction block.

The exact flow in MACE-MP-0 is:
1. `models.py: MACE.forward` receives `positions`.
2. `prepare_graph()` computes exact `vectors` and `lengths` (this is the edge geometry).
3. `models.py:321` computes `edge_attrs = self.spherical_harmonics(vectors)` (angular representation).
4. `models.py:322` computes `edge_feats, cutoff = self.radial_embedding(lengths, ...)` (radial representation).
5. The loop at `models.py:363` calls the `InteractionBlock`. For MACE-MP-0, this is typically `RealAgnosticResidualInteractionBlock` (or similar subclass in `blocks.py`).
6. Inside `InteractionBlock.forward` (`blocks.py:763`), `tp_weights = self.conv_tp_weights(edge_feats)` is computed.
7. **The Cacheable Unit:** At `blocks.py:791`, the Clebsch-Gordan tensor product is computed for all edges:
   ```python
   mji = self.conv_tp(node_feats[edge_index[0]], edge_attrs, tp_weights)
   ```
   For `first_layer = True`, `node_feats` is purely the species embedding (zero closure). Therefore, `mji` shape `[n_edges, irreps]` is perfectly equivalent to the Frost tensor $g_{ij}$.
8. `blocks.py:794`: `mji` is immediately aggregated over edges via `scatter_sum(src=mji, index=edge_index[1], ...)` into `message` for each node.
9. Deeper MACE layers continue using the `message` to update node features.

## C. What can actually be intercepted

We must replace `mji` inside `RealAgnosticResidualInteractionBlock.forward` for the selected "clean" edges *before* `scatter_sum` is called.

The smallest intervention is to dynamically monkey-patch the `forward` method of the `InteractionBlock` class that MACE instantiates for its first layer.

We can wrap `MACE.interactions[0].forward` with a custom function:
```python
original_forward = mace_model.interactions[0].forward

def patched_forward(self, *args, **kwargs):
    # Call original forward up to the tp_weights calculation
    # Intercept `conv_tp` computation
    # ...
```
Actually, it's easier to dynamically subclass the instantiated interaction block or replace the `conv_tp` module call temporarily, or simply substitute `InteractionBlock.forward` for the duration of the context manager.

By intercepting `mji = self.conv_tp(...)`, we can inject:
- **Mode 1:** `mji[clean] = g_ref[clean]`
- **Mode 2:** `mji[clean] = g_ref[clean] + einsum(delta_r, J_ref)`
and then pass the substituted `mji` directly into the `scatter_sum` aggregation, leaving the rest of MACE (layers 2+) completely untouched.

## D. Repair status

The implementation now intercepts `mji` inside the first unfused MACE
interaction block. Cached evaluations keep `Atoms` coordinates unchanged and
substitute only selected edge messages. The first-order path uses
`g_ref + J_ref @ delta_r`, with `J_ref` extracted by PyTorch autograd from
MACE's internal vectors and edge ordering.

The seven-case integrity matrix is implemented in
`frost/e1/force_error_tests.py` and passes with the original tests (15 total).
New or disappearing edges invalidate alignment and are treated as dirty.

The reusable model-facing API is ready, but the long trajectory callback still
uses an initial layer-1 reference snapshot. Per-edge, per-tolerance refresh of
`g_ref` and `J_ref` must be added before production trajectory numbers are
interpreted as the final Frost result.
