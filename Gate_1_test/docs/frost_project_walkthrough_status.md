# Frost Project Walkthrough and Status Briefing

**Status date:** 2026-09-04  
**Scope:** Frost specification/roadmap and the current E1 repository state

## Executive Summary

Frost is an attempt to accelerate molecular dynamics driven by geometric graph
neural networks. The observation is simple: between adjacent MD steps, most
local pair geometries change only slightly, but the model recomputes expensive
edge geometry for every edge anyway. Frost proposes caching those edge-local
intermediates and recomputing only the edges whose geometry has changed enough
to matter.

The important correction in this project is that Frost must cache an internal
edge representation, not atomic coordinates. The original experiment froze
whole atoms at old coordinates and therefore measured the force error of an
artificial, strained structure. That result could not test Frost. The repaired
unit-level method intercepts MACE's first-layer edge message `mji` before it is
aggregated, while leaving the current atomic configuration untouched.

What is proven today is narrower than "Frost works": the edge-local
substitution mechanism, stable ASE/MACE edge identity, exact-control force path,
and a tiny no-Jacobian MD canary have passed. The long E1 pilot, first-order
trajectory path, and all roadmap gates remain unvalidated.

## 1. What Frost Is Trying to Achieve

An MD simulation repeatedly advances a configuration of atoms:

$$R_t \longrightarrow R_{t+1}.$$

At each step, an interatomic potential predicts an energy and forces. A
geometric GNN represents the atoms as nodes and nearby atom pairs as directed
edges. The expensive part is often an edge-wise geometric calculation followed
by message aggregation and node updates.

Frost exploits temporal redundancy. If edge $(i,j)$ has almost the same local
geometry now as when it was last evaluated, its expensive geometric message can
be reused or locally approximated. A changed edge is called **dirty** and is
recomputed. An unchanged edge is **clean** and may use its cache entry.

This is useful because the decision is local to an edge. A single moving atom
does not automatically invalidate every node in its multi-hop neighborhood. The
PDF calls this the failure of node-level caching: after several message-passing
layers, a dirty node's graph closure can saturate most of the system. Edge-level
caching avoids that closure at the first geometric layer.

The intended benefit is reduced tensor-product work, not reduced physical
fidelity by moving or freezing atoms. The exact current structure remains the
input to the model; only selected intermediate edge tensors are reused.

## 2. Technical Concepts

### 2.1 Edge geometry

For a directed edge from atom $i$ to atom $j$, the basic geometric quantities
are:

- $r_{ij}$: the displacement vector from $i$ to $j$, including the periodic
  image shift and minimum-image convention.
- $d_{ij}=\|r_{ij}\|$: the scalar distance.
- $\hat r_{ij}=r_{ij}/d_{ij}$: the unit direction.
- `(i, j, shift)`: the cache key. The periodic shift is necessary because the
  same atom pair can have multiple periodic-image edges.

The current geometry code builds these values from the ASE neighbor list. It
does not use a naive Cartesian difference that would ignore periodic boundary
conditions.

### 2.2 Geometric GNNs, MACE, and Allegro

A geometric GNN must respond correctly to translations, rotations, and often
reflections. MACE and Allegro achieve this with equivariant features: their
internal feature channels transform in known ways under rotation rather than
being ordinary scalar vectors.

**MACE-MP-0** is the pretrained model currently available in this repository.
**Allegro** is another relevant architecture, but the repository does not have
an appropriate universal checkpoint, so Allegro is currently blocked for E1.

At a high level, MACE does the following:

1. Builds a neighbor graph from positions, cell, PBC, and cutoff.
2. Converts each edge's distance and direction into radial and angular features.
3. Combines source-node features with those edge features using an equivariant
   tensor product.
4. Aggregates edge messages into destination nodes.
5. Repeats message passing through deeper layers and produces energy and forces.

### 2.3 Layer 1 and `mji`

The first MACE interaction block is special. Its source node representation is
essentially the species embedding, so the first edge tensor product is a
particularly clean edge-local target for caching.

Inside the first interaction block, MACE computes:

```text
mji = conv_tp(source_node_features, edge_attrs, tensor_product_weights)
```

`mji` is one message per directed edge. Immediately afterward, MACE performs a
`scatter_sum` over destination atoms. The repaired implementation intercepts
`mji` between those two operations. This is the controlling abstraction
boundary: the cache changes edge messages, while MACE still performs the normal
aggregation and all later computation.

### 2.4 `g_ref`, $J_{ref}$, and cache modes

`g_ref` is the cached first-layer edge message at an edge's reference geometry.
For a current edge vector $r$ and reference vector $r_{ref}$, the first-order
cache uses:

$$\hat g(r)=g_{ref}+J_{ref}(r-r_{ref}),$$

where

$$J_{ref}=\frac{\partial g}{\partial r}\bigg|_{r_{ref}}.$$

The implementation extracts $J_{ref}$ with PyTorch autograd from MACE's
internal edge messages and vectors.

There are two cache modes:

- **Zeroth order:** use $\hat g=g_{ref}$. This is retained as a negative
  control. The PDF's force-error analysis says it is inadmissible as the final
  design because its force contribution can be wrong even when the cached
  message value appears close.
- **First order:** use $\hat g=g_{ref}+J_{ref}\Delta r$. This is the minimum
  scientifically admissible Frost candidate in the revised roadmap.

Cached tensors must be detached from the current autograd tape in a production
implementation. That conserves the intended force path and avoids retaining
unnecessary activations.

### 2.5 Dirty edges, refresh, $\phi$, and $\theta$

For tolerance $\epsilon$, the invalidation predicate compares current and
reference edge geometry. The current E1 code uses a geometric displacement and
direction test with a unit sensitivity factor. A dirty edge is refreshed; a
clean edge retains its reference state.

The PDF uses:

- $\phi(\epsilon)$: the dirty fraction, approximately
  $|D|/|E|$. Lower is better because fewer edge tensor products need to be
  recomputed.
- $\theta$: the cacheable work share,
  $\theta=c_{tp}/(c_{tp}+c_{agg})$. It belongs primarily to E2's cost
  decomposition and the speedup model, not to the geometric definition of
  dirty edges.

The intended final scheduler also handles neighbor-list rebuilds, new and
removed edge keys, maximum cache age, keyframes, and a dense fallback when too
many edges are dirty. Those are roadmap mechanisms, not all current E1
features.

## 3. What E1 Is Supposed to Do

According to the PDF, E1 is the first characterization experiment and the
first decision gate. Its chronological job is:

1. Select the benchmark systems, temperatures, model, and reproducible seeds.
2. Run NVT Langevin MD with a 1 fs timestep and 0.01 fs^-1 friction.
3. Equilibrate for 20 ps, discard those frames, then record 10,000 production
   steps.
4. At every production step, build the edge list and measure current-versus-
   reference perturbations.
5. Sweep the specified tolerances from 0.1 through 50 meV/A.
6. Maintain independent per-edge reference state and refresh dirty edges.
7. Measure $\phi(\epsilon)$, edge lifetimes, perturbation distributions, and
   the joint relationship between perturbation and force error.
8. On sampled frames, evaluate exact and cached forces on the identical current
   $R_t$, with edge-level message substitution only.
9. Compare zeroth- and first-order cache behavior, including force-error
   scaling.
10. Apply Gate 1: the PDF's original practical criterion is roughly
    $\phi<0.5$ at a defensible tolerance on favorable systems. A negative result
    is still a valid characterization result, but it changes the project claim.

The planned full matrix is MACE on five systems, two temperatures, and three
seeds. The current configuration contains the 20 ps/10,000-step protocol and
eight tolerances; its pilot is only 100 equilibration and 500 production steps.

## 4. What Was Originally Implemented and Why It Failed

The original force-error path classified edges and then converted that result
into a set of stale atoms. It replaced selected atoms' current coordinates with
reference coordinates and ran MACE on the resulting mixed-time structure.

That is not equivalent to Frost. If atom $i$ is moved backward while a neighbor
$j$ remains current, the resulting $r_{ij}$ is a new, physically strained
configuration. MACE correctly computes forces for that artificial structure,
but those forces measure coordinate tampering, not stale edge-message reuse.
The large force error therefore could not support a claim that Frost fails.

The repair removed the scientific dependency on atom freezing. Exact and cached
evaluations now use the same coordinates, cell, and PBC. Only selected layer-1
`mji` messages differ. New or disappearing edges are treated as invalid rather
than silently aligned by array position.

## 5. What the Current Code Demonstrates

The relevant surfaces are:

- [mace_patch.py](../frost/e1/mace_patch.py): temporarily patches the first
  unfused MACE interaction block and substitutes edge messages before
  aggregation.
- [force_error.py](../frost/e1/force_error.py): extracts layer-1 messages and
  Jacobians, computes exact forces, and provides cached-force evaluation.
- [reference_cache.py](../frost/e1/reference_cache.py): tracks edge keys,
  geometry references, age, generation, validity, dirty masks, and new/removed
  edges.
- [trajectory.py](../frost/e1/trajectory.py): supplies the NVT Langevin MD
  driver.
- [__main__.py](../frost/e1/__main__.py): stages trajectory logging and sampled
  force comparisons, but is not yet a validated final E1 controller.

The focused differential matrix covers exact substitution, stale-edge
zeroth-order error, first-order improvement, zero displacement, expected
scaling, multiple clean edges, and periodic geometry. The earlier checkpoint
reported 15 passing tests. After adding the exact-mode regression exposed by
the trajectory attempt, the current focused command reports **16 passed**.
These are integrity tests for the interception and bookkeeping behavior; they
are not evidence of a completed MD experiment or speedup.

## 6. What the Current Short Trajectory Did

The short standalone check loads MACE-MP-0, constructs diamond Si, initializes
300 K Langevin MD, and attempts five equilibration steps plus three production
steps. Its callback builds the ASE edge table, advances the per-edge cache, and
evaluates exact versus cached forces on production frames.

The earlier Jacobian-containing probe became computationally pathological and
eventually terminated without a scientific result. It computed a separate
autograd gradient for every edge-message feature while retaining the graph.
That is an expensive way to materialize `[edges, features, 3]`, especially in
float64.

The replacement canary deliberately omitted Jacobians. It completed three
production steps after two equilibration steps on Si at 300 K. Every step had
2,944 edges, no additions or removals, all edges clean at the tested
tolerances, unchanged coordinates, finite values, and exact-control force error
below $1.1\times10^{-15}$ eV/A.

## 7. Roadmap Status

### Complete or demonstrated

- Environment and MACE-MP-0 loading have been verified in `frost_e1`.
- The atom-freezing methodology failure has been identified and documented.
- The correct MACE layer-1 interception point has been located.
- Zeroth- and first-order edge-message substitution works in the focused
  differential harness.
- Periodic edge geometry and edge-key handling have focused tests.
- Per-edge reference metadata, generations, ages, and dirty bookkeeping exist.
- A checkpoint commit exists at `43d5bd7`.

### Partially complete

- Per-edge trajectory state exists, but the cache values `g_ref` and `J_ref`
  are not yet fully maintained independently for every tolerance and edge in a
  validated trajectory.
- The trajectory callback is still scaffolding. It has not demonstrated that
  geometry-cache keys map correctly to MACE's runtime edge ordering across
  frames.
- The HDF5 logger and plotting pipeline exist, but current artifacts are from
  the earlier invalid/staged experiment and must not be read as repaired E1
  results.

### Not validated

- A completed post-repair first-order short MD trajectory.
- Exact-versus-first-order-cached force errors on real MACE trajectory frames.
- Stable edge counts and neighbor-list remapping during trajectory execution.
- No-coordinate-mutation and identical-$R_t$ checks in a completed run.
- Full 20 ps plus 10,000-step E1 pilot.
- The five-system, two-temperature, three-seed E1 matrix.
- Gate 1's $\phi$ decision.
- E2 cost decomposition and measured $\theta$.
- Any production Frost speedup, deep-layer closure result, drift result, or
  Allegro result.

## 8. Chronological Next Steps

1. Finish or formally terminate the currently running short probe and preserve
   its complete stdout/stderr and exit status. Do not interpret its lack of
   output as success.
2. Make the cheapest validation probe capable of completing: first exercise
   the edge-cache state machine without Jacobian extraction, then run a minimal
   MACE force comparison with a precomputed, explicitly checked edge-order
   mapping. This is validation work, not an optimization claim.
3. Run a repaired short trajectory and record, for every step: edge count,
   edge-key changes, dirty and clean fractions per tolerance, refresh and
   generation events, coordinate hashes before/after cache evaluation, exact
   versus cached force errors, and finite-value checks.
4. Only after that passes, run the configured MACE Si 300 K pilot with its
   actual pilot protocol and inspect the HDF5 data and plots.
5. Run the full E1 matrix: MACE, five systems, 300/1000 K, three seeds, 20 ps
   equilibration, 10,000 production steps, and 100 sampled force-error frames.
   Allegro remains blocked until a valid checkpoint exists.
6. Apply Gate 1 using measured $\phi(\epsilon)$ together with the measured
   perturbation-force-error distribution. Decide whether Frost remains a
   systems acceleration project or becomes a characterization paper.
7. Proceed to E2: instrument the model to measure tensor-product versus
   aggregation and predicate costs, determine $\theta$ and the break-even
   regime, and decide the default first-order representation.
8. Continue to the later roadmap stages only after E1/E2: cache manager and
   fused predicate, sparse dispatch, Allegro end-to-end, E5/E8 speed and stride
   studies, conservativity and drift (E7), attenuated deep-layer closure (E3/
   E4), scale/memory, fidelity, adversarial/precision/cross-generation tests,
   and finally control-loop and cross-domain experiments.

## 9. Specification vs Demonstration vs Hypothesis

| Category | Statements currently supported |
|---|---|
| **Frost PDF says** | Temporal edge-geometry redundancy may provide useful savings; node-level closure is a problem; first-order caching is the minimum admissible force-aware design; E1 measures $\phi$ and perturbation/force-error behavior; E2 measures cost share $\theta$; later gates determine whether to continue. |
| **Our implementation demonstrates** | MACE's first-layer `mji` can be intercepted before scatter aggregation; exact mode leaves messages unchanged; edge-local zeroth- and first-order substitutions behave as designed in focused tests; periodic edge geometry and basic cache lifecycle bookkeeping pass tests. |
| **Still a hypothesis** | Real trajectories will have a sufficiently low dirty fraction; first-order cached forces will remain within the intended tolerance; the current cache metadata will align with MACE runtime edges over neighbor-list changes; the cache will produce useful speedup; deep-layer treatment, conservation, and Allegro behavior will work. |
| **Explicitly not a result** | The old atom-freezing FAIL verdict, the existing plots/HDF5 artifact from that path, and the still-running Jacobian-heavy short probe are not valid repaired E1 scientific conclusions. |

The project is therefore at the boundary between a validated mechanism and an
unvalidated experiment. The next scientifically meaningful milestone is not a
full run: it is one completed, instrumented short trajectory proving that the
edge-level implementation preserves the current structure and produces a
traceable exact-versus-cached comparison.