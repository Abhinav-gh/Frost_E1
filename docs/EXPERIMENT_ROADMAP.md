# Experiment roadmap for PASC 2027

## Evidence question

Measure the gap between (1) geometric edge stability, (2) error from reusing edge messages, and (3) end-to-end performance after cache and derivative-action overhead. The result can be a negative result if that gap is measured carefully, the implementation path is audited, and the conclusions are scoped to evaluated models, systems, and hardware.

## Workstreams and gates

1. **W0 — implementation audit (required first).** On saved configurations, compare exact and substituted edge messages, clean/dirty behavior, edge ordering, storage aliasing, and force/energy outputs. Include zero displacement, controlled single-atom perturbations, cache disabled, normal reuse, and deliberately all-clean diagnostics. Resolve any mismatch before calling later values cache accuracy.
2. **W1 — snapshot accuracy frontier.** Use 50–100 time-stratified snapshots per trajectory across a displacement-tolerance range. Record clean fraction, force RMS/max and per-atom tails, energy error, and reference age. Compare zeroth- and first-order estimates. Report distributions, temporal block uncertainty, and error-versus-reuse plots.
3. **W2 — trajectory replication and temperature.** Run at least three independent trajectories for the main material/temperature, then compare 100/300/600 K if stable. Treat trajectory as the replication unit; snapshots are correlated.
4. **W3 — scale/material scope.** Evaluate increasing Si supercells and, if the pretrained potential is suitable, a second material/model. State model-specific limitations explicitly.
5. **E4/W4 — full cost accounting.** Run this after the core frontier and before broad generalization. Benchmark fixed configurations with warmups, repeated synchronized complete-calculator evaluations, and exact/disabled/dirty-only/controlled/natural/first-order arms. Separately report complete MD wall time. Include full component cost and a break-even analysis; do not use isolated layer timing as the speedup claim.
6. **E5/W5 — rollout behavior.** Run the short rollout after E4/W4 validation, then the longer rollout. Compare exact and cache MD from identical equilibrated initial positions and momenta. Track temperature, energy, structural observables, errors on cache-produced frames, and failures. Because trajectories can diverge chaotically, compare ensemble/distribution behavior rather than demanding long-time pointwise identity.
7. **W6 — optional policies.** Only after fixed-threshold evidence is stable, evaluate stride/adaptive refresh calibrated on training trajectories and held-out trajectories. Do not present a heuristic as a certified error bound.

## Decision rules

- Stop and repair if W0 shows edge-order mismatch, dirty edges do not match exact messages, or the exact/zero-displacement control fails.
- If clean fraction rises while force error exceeds a predeclared budget, report that as an accuracy/reuse tradeoff; do not select tolerance post hoc to imply a safe operating point.
- If complete calculator/MD runtime remains slower, quantify where time goes and report the negative performance outcome. It can still support a systems characterization if the overhead decomposition and measured scope are reproducible.
- Prioritize W0, W1, W2 at the primary temperature, and W4. Add one temperature or size result only if the core pipeline is validated and the GPU budget allows. Treat second material and adaptive refresh as stretch goals.

## Claim boundaries

Current reuse/error measurements and the reported large integrated slowdown are pilot observations, not universal claims. Do not describe a clean fraction as a cacheable force fraction; do not infer force accuracy from zero measured errors on a handful of frames; do not generalize MACE-MP-0 silicon results to other models or materials.
