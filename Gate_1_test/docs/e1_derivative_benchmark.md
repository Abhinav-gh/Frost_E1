# E1 First-Order Derivative Decision Record

**Date:** 2026-09-07  
**Scope:** derivative required for the first-order layer-1 message update and conservative force calculation

## Current MACE Shapes

On the fixed 64-atom diamond-Si configuration with MACE-MP-0 medium, the current capture path reports:

- edge count: `E = 2944`
- layer-1 message `g`/`mji`: `(E, F) = (2944, 2048)`
- edge vectors `r`: `(E, 3)`
- edge index: `(2, E) = (2, 2944)`
- dtype/device: `float64`, CUDA

For one edge:

$$r_e \in \mathbb{R}^{3}, \qquad g_e(r_e) \in \mathbb{R}^{F}, \qquad F=2048.$$

Because the first-layer source feature is fixed by species and the edge radial/angular inputs are edge-local, the derivative is block diagonal over edges:

$$J_e = \frac{\partial g_e}{\partial r_e} \in \mathbb{R}^{F\times 3}.$$

There is no cross-edge derivative in this first-layer message map under the current MACE path.

## Mathematical Requirements

### Message update

The first-order approximation is:

$$\hat g_e(r_e+\Delta r_e)=g_{e,ref}+J_{e,ref}\Delta r_e+O(\|\Delta r_e\|^2).$$

The required linear operation for one known displacement is therefore a **JVP**:

$$J_{e,ref}\Delta r_e \in \mathbb{R}^{F}.$$

A full `J_ref` is not mathematically required merely to construct one message update if the implementation can evaluate the JVP directly.

### Conservative force from approximated energy

Let the downstream MACE energy be $E_{down}(\hat g,\ldots)$ and define:

$$u_e = \frac{\partial E_{down}}{\partial \hat g_e}\in\mathbb{R}^{F}.$$

Since $J_{e,ref}$ is treated as a fixed cached derivative, differentiating the affine message with respect to the current edge vector gives:

$$\frac{\partial E_{hat}}{\partial r_e}=J_{e,ref}^{T}u_e\in\mathbb{R}^{3}.$$

The force contribution is:

$$F_e=-J_{e,ref}^{T}u_e.$$

This is a **VJP/contraction**, not another full Jacobian requirement. The downstream model supplies $u_e$ during backpropagation; the cache must supply the action of $J_{e,ref}^{T}$ on that upstream vector.

### What must be stored or recomputed

There are three implementation choices:

1. Store the dense per-edge `J_ref` with shape `(E,F,3)`. This supports both the forward JVP and backward VJP with cheap contractions, but costs substantial memory.
2. Store a compressed linear operator/factorization that supports both `J @ delta` and `J.T @ upstream`. This is the eventual systems option, but is outside the current validation scope.
3. Avoid storing `J_ref` and recompute JVP/VJP actions from a reference derivative graph. This may reduce storage but can reintroduce expensive derivative computation and must be benchmarked against the actual MACE graph.

A JVP alone is sufficient for a detached message value, but not sufficient by itself for conservative forces unless the backward VJP action is also available. A VJP alone is insufficient to form the forward affine update for an arbitrary `delta_r`.

## Existing Pathology

The current `extract_layer1_reference_cache` materializes the dense Jacobian by looping over every edge and every message feature:

```python
for edge in range(messages.shape[0]):
    for feature in range(messages.shape[1]):
        autograd.grad(messages[edge, feature], vectors, retain_graph=True)
```

For the actual Si shape this is approximately:

$$2944\times2048=6,029,312$$

individual gradient calls. Each call retains the common graph. This is why the previous probe became computationally pathological. It is not a scientific result.

Dense output storage alone for float64 is approximately:

- messages `(E,F)`: `48,234,496` bytes, about `46 MiB`
- dense Jacobian `(E,F,3)`: `144,703,488` bytes, about `138 MiB`
- one JVP result `(E,F)`: about `46 MiB`
- one VJP force contraction `(E,3)`: `70,656` bytes, about `0.067 MiB`

These are output sizes only. Retained autograd graph memory can be much larger.

## Isolated Benchmark

The benchmark is [derivative_benchmark.py](../frost/e1/derivative_benchmark.py). It uses a tiny independent edge-local map from `R^3` to `R^12`, with `E=8` edges, and compares every candidate against a reference Jacobian.

Reference shapes:

- full Jacobian: `(8, 12, 3)`
- JVP message update: `(8, 12)`
- VJP force contraction: `(8, 3)`

Measured results from the `frost_e1` environment:

| Method | Output shape | Runtime (s) | Max difference | Output storage |
|---|---:|---:|---:|---:|
| Existing elementwise `autograd.grad` loop | `(8,12,3)` | `0.026621369` | `0` | 2304 bytes |
| `vmap(jacrev)` full Jacobian | `(8,12,3)` | `0.001901196` | `0` | 2304 bytes |
| `vmap(jacfwd)` full Jacobian | `(8,12,3)` | `0.099011181` | `0` | 2304 bytes |
| Batched JVP | `(8,12)` | `0.001793876` | `1.735e-18` | 768 bytes |
| Batched VJP | `(8,3)` | `0.001331528` | `4.441e-16` | 192 bytes |

The benchmark is intentionally small and measures eager CPU execution. The
transform timings include PyTorch transform behavior and should not be treated
as MACE performance numbers. The robust conclusions are the shapes and the
exact numerical agreement, not the absolute seconds.

## Recommendation

Do not change MACE integration yet. The mathematically correct target is:

- forward first-order message: JVP action `J_ref @ delta_r`;
- conservative backward force: VJP action `J_ref.T @ upstream`;
- dense `J_ref` is only one representation that provides both actions.

The smallest next MACE experiment should use one tiny fixed configuration and
one deliberately small edge/message slice or toy wrapper around the captured
MACE message. It should compare:

1. dense `vmap(jacrev)` JVP and VJP contractions;
2. the proposed direct JVP/VJP action path;
3. the resulting energy and force gradients through the affine message.

It must verify both the forward message update and `F=-∇E` before considering
any full 2048-feature, 2,944-edge integration. No trajectory and no CUDA
optimization are required for that experiment.

## Real-MACE Reference Validation

The real MACE local edge function was reconstructed from the first interaction
block: spherical harmonics, radial embedding and cutoff weights, tensor-product
weights, and `conv_tp` applied to the fixed transformed source-node feature.
For one captured edge, it reproduces MACE's intercepted `mji` message with
maximum difference `2.22e-15`.

`extract_layer1_dense_jacobian` in `frost/e1/force_error.py` uses
`vmap(jacrev)` over this edge-local map and returns `(E_subset, 2048, 3)`.
Eight edges completed in `0.86 s`, with finite CUDA float64 output and message
reproduction difference `2.84e-14`.

An independent `autograd.functional.jacobian` calculation on four real MACE
edges agreed with the vectorized result:

- shape: `(4, 2048, 3)`;
- maximum absolute difference: `2.61e-14`;
- mean absolute difference: `3.21e-16`;
- independent runtime: `30.6 s`.

The installed e3nn spherical-harmonics TorchScript kernel allows one
`vmap(jacrev)` call of about 16 edges, but a second transformed call fails with
`Cannot access data pointer of Tensor that doesn't have storage`. Rebuilding
the transform and using the functionally equivalent e3nn API do not remove this
limitation. The dense oracle is therefore bounded to a small validated subset
rather than silently starting a multi-hour full-system extraction.

### Direct derivative actions

The implementation now exposes two isolated helpers in
`frost/e1/force_error.py`:

- `layer1_edge_jvp(...)`: one-edge `autograd.functional.jvp`, returning
  `(2048,)` for a direction in `R^3`;
- `layer1_edge_vjp(...)`: one-edge ordinary autograd contraction, returning
  `(3,)` for an upstream vector in `R^2048`.

These are deliberately one-edge operations. `torch.func.jvp` and
`torch.func.vjp` trigger the installed e3nn TorchScript storage failure because
the transform introduces dual tensors that the spherical-harmonics kernel
cannot consume. The functional autograd forms avoid that incompatibility
without changing MACE or the cache.

On the 16-edge real-MACE diagnostic, direct JVP versus dense-Jacobian JVP had
maximum differences of `5.20e-18`, `4.86e-17`, and `4.44e-16` at displacements
`1e-4`, `1e-3`, and `1e-2 A`, respectively. Direct VJP versus dense-Jacobian
VJP had maximum difference `3.92e-14`.

The direct actions are numerically equivalent to the dense reference on the
validated subset. They are a correctness implementation, not yet a production
performance implementation.

### JVP / message test

Using 16 real MACE edges and a controlled displacement of atom 0:

| displacement | max message error | mean error | RMS error | relative norm |
|---:|---:|---:|---:|---:|
| `1e-4 A` | `1.775e-6` | `2.272e-8` | `9.462e-8` | `1.508e-8` |
| `1e-3 A` | `1.777e-4` | `2.272e-6` | `9.462e-6` | `1.508e-6` |
| `1e-2 A` | `1.788e-2` | `2.272e-4` | `9.462e-4` | `1.508e-4` |

The approximately quadratic residual confirms first-order Taylor behavior.

### VJP test

For four real MACE edges and a deterministic upstream tensor with shape
`(4, 2048)`, both the dense contraction and independent autograd returned
`(4, 3)`:

- maximum absolute difference: `3.92e-14`;
- mean absolute difference: `1.07e-14`;
- relative norm difference: `2.25e-15`.

### Actual first-order Frost interception

The actual MACE interception was run with the first 16 edges using the dense
Jacobian reference. The other 2,928 edges remained exact, so this is a
controlled partial-edge experiment rather than a full dense-cache run.

For a `1e-3 A` displacement:

- exact energy: `-341.829765835653 eV`;
- Frost energy: `-341.829766148569 eV`;
- absolute energy error: `3.129e-7 eV`;
- maximum force error: `6.258e-4 eV/A`;
- mean force error: `1.481e-5 eV/A`;
- RMS force error: `7.332e-5 eV/A`;
- relative force norm error: `1.040e-1`;
- exact force norm: `9.766e-3 eV/A`;
- Frost force norm: `9.025e-3 eV/A`;
- all values finite: yes.

For zero displacement, the same path produced zero energy error and maximum
force error `9.14e-16 eV/A`. The relative force metric is not meaningful
there because the exact force norm is approximately zero.

## Status Boundary

Proven:

- the current MACE shapes are `(2944,2048)` messages and `(2944,3)` vectors;
- the derivative is edge-local with per-edge shape `(2048,3)`;
- message construction requires a JVP action;
- conservative force recovery requires a VJP action;
- the isolated candidates agree numerically on the toy map;
- `vmap(jacrev)` avoids the pathological per-element loop on the toy map.
- the real-MACE edge-local message function reproduces captured `mji`;
- the real-MACE dense Jacobian agrees with independent autograd on four edges;
- real-MACE JVP, VJP, partial-edge Frost, and zero-displacement controls pass.

Not yet proven:

- a practical full-system dense extraction under the installed e3nn transform
  limitation;
- whether direct MACE JVP/VJP actions preserve the intended detached-cache
  semantics and force path;
- whether a dense, compressed, or recomputed representation is best in the
  final runtime.

## Full-System First-Order Result

The dense reference was then run on all 2,944 MACE edges. The derivative oracle
uses the exact e3nn low-order polynomial formulas in eager Torch form for the
derivative-only path; these match MACE's scripted spherical harmonics with zero
observed difference on a 32-vector comparison. Normal MACE and cache execution
remain unchanged.

Jacobian result:

- shape: `(2944, 2048, 3)`;
- dtype/device: `float64`, CUDA;
- finite: yes;
- runtime: `5.23 s` in the instrumented run;
- output bytes: `144,703,488`;
- peak CUDA allocation: `1,886,232,064` bytes.

The controlled perturbation moves atom 0 in the positive x direction by the
listed amount. All 2,944 edge messages are replaced by
`g_ref + J_ref @ delta_r` in the Frost path.

| displacement | message RMS error | energy error (eV) | force max error (eV/A) | force RMS error (eV/A) | relative force error |
|---:|---:|---:|---:|---:|---:|
| `0` | `2.173e-16` | `0.000e+00` | `6.891e-16` | `2.623e-16` | `1.221`* |
| `1e-4 A` | `1.404e-7` | `1.409e-8` | `2.819e-4` | `2.409e-5` | `3.418e-1` |
| `1e-3 A` | `1.404e-6` | `1.409e-6` | `2.819e-3` | `2.409e-4` | `3.418e-1` |
| `1e-2 A` | `1.404e-4` | `1.410e-4` | `2.821e-2` | `2.411e-3` | `3.421e-1` |

`*` The zero-displacement relative force ratio is not meaningful because the
exact force norm is approximately zero (`2.98e-15 eV/A`).

Detailed full-system energy values:

- `0 A`: exact `-341.829769930151`, Frost `-341.829769930151` eV;
- `1e-4 A`: exact `-341.829769889206`, Frost `-341.829769903300` eV;
- `1e-3 A`: exact `-341.829765835653`, Frost `-341.829767245102` eV;
- `1e-2 A`: exact `-341.829360485206`, Frost `-341.829501482002` eV.

Observed scaling exponents between `1e-4` and `1e-3 A`, then between `1e-3`
and `1e-2 A`:

- message RMS error: `2.0000`, `2.0000`;
- energy error: `2.0000`, `2.0002`;
- force RMS error: `1.0000`, `1.0005`.

This supports the expected first-order behavior: message and energy residuals
are quadratic in displacement, while force residuals are linear because the
cached Jacobian is held fixed during force differentiation. All full-system
values were finite. No MD or tolerance invalidation was involved.
