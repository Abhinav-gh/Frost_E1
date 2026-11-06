# Frost Project: Research-Grade Codebase Upgrade, Optimization, and Experimental Infrastructure

## ROLE

You are acting as a senior research software engineer, scientific computing engineer, ML systems engineer, and performance optimization specialist.

Your task is to substantially improve the Frost edge-caching research codebase so that it is suitable for rigorous scientific experimentation, reproducible benchmarking, and eventual publication.

This is NOT merely a code cleanup task.

You must:
- Understand the existing research implementation.
- Audit its mathematical and computational correctness.
- Implement the missing optimizations and improvements described below.
- Improve software architecture and maintainability.
- Build robust experimental and analysis infrastructure.
- Add comprehensive testing.
- Update documentation and configuration files.
- Make the repository ready for systematic research campaigns.

You are expected to IMPLEMENT the improvements, not merely produce a proposal.

---

# 1. PROJECT CONTEXT

Frost investigates temporal reuse of geometric message-passing computations in equivariant graph neural networks, particularly MACE interatomic potentials.

The fundamental idea is to reuse edge-local messages across successive molecular dynamics configurations when the geometry associated with an edge has not changed sufficiently.

The current implementation focuses on Layer-1 MACE messages, particularly the edge-local message computation before scatter aggregation.

The cache operates on reference edge geometry and distinguishes between:
- Clean edges: eligible for reuse.
- Dirty edges: require recomputation.

The implementation has explored:

### Zeroth-order caching
\[
g(r_t) \approx g(r_{\mathrm{ref}})
\]

### First-order caching
\[
g(r_t) \approx g(r_{\mathrm{ref}})
+ J_{\mathrm{ref}}\Delta r
\]

where:
- \(g\) is the edge-local message function.
- \(J_{\mathrm{ref}}\) is the Jacobian with respect to the edge displacement.
- \(\Delta r\) is the displacement relative to the reference geometry.

The project also investigates:
- Temporal geometric redundancy.
- Edge reuse versus force-error tradeoffs.
- Matrix-free Jacobian-vector and vector-Jacobian products.
- GPU execution efficiency.
- End-to-end inference performance.
- MD rollout behavior.
- Generalization across trajectories, temperatures, and system sizes.

The research objective is to characterize the actual opportunities and limitations of temporal edge reuse, not to assume that caching necessarily produces acceleration.

---

# 2. IMPORTANT PROJECT RULES

## 2.1 Preserve scientific integrity

Never modify the mathematical approximation, cache semantics, force computation, or experiment protocol silently.

If a modification changes the scientific method:
1. Identify the change.
2. Explain its mathematical consequences.
3. Add appropriate tests.
4. Record it in documentation.
5. Preserve the previous implementation when needed for comparison.

Do not optimize away computations required for correctness.

Do not use:
- Artificially selected snapshots.
- Cherry-picked tolerances.
- Unreported cache states.
- Inconsistent exact/Frost starting configurations.
- Invalid speedup measurements.
- Silent fallback implementations.
- Fake or fabricated experimental results.

## 2.2 Preserve existing results

The existing legacy results are important research artifacts.

Do not delete, overwrite, or reorganize existing legacy outputs.

Use:
- `results/legacy_baseline/` for preserved historical results.
- `results/active/` for new experiments.
- New unique experiment directories for all new executions.

Never overwrite previous results unless explicitly authorized.

## 2.3 Understand before modifying

Before implementing anything:
- Inspect the repository.
- Inspect all existing source modules.
- Inspect tests.
- Inspect experiment scripts.
- Inspect configurations.
- Inspect the project roadmap and progress documentation.
- Identify existing functionality that already satisfies requirements.

Do not duplicate existing functionality unnecessarily.

## 2.4 Use the current implementation as the source of truth

Do not assume that the repository matches an old README or previous conversation exactly.

Verify:
- Actual function signatures.
- Current CLI options.
- Cache modes.
- Existing experiment outputs.
- Implemented batching.
- Existing correctness tests.
- CUDA behavior.
- Current dependencies.

If previous documentation contradicts the implementation, identify the discrepancy and resolve it carefully.

---

# 3. PHASE 0 — REPOSITORY RECONNAISSANCE AND ARCHITECTURE AUDIT

Begin by examining the entire repository.

At minimum, inspect:

- `frost/e1/`
- `experiments/`
- `configs/`
- `tests/`
- `docs/`
- `results/`
- `pyproject.toml`
- Dependency and installation files.
- Existing README files.
- Existing benchmark and analysis modules.

Produce an internal implementation map covering:

1. Cache state management.
2. Reference geometry handling.
3. Edge ordering and edge-to-reference mapping.
4. MACE message extraction and patching.
5. Exact versus approximate inference paths.
6. Jacobian extraction.
7. JVP/VJP implementations.
8. Batched action implementations.
9. Stage A execution.
10. Stage B execution.
11. E3 audit.
12. Campaign execution.
13. Generalization experiments.
14. Inference benchmarking.
15. End-to-end benchmarking.
16. MD rollout experiments.
17. Existing analysis and visualization.
18. Testing and CI.

Identify:
- Duplicate code.
- Dead code.
- Unused imports.
- Hidden assumptions.
- Hardcoded values.
- Incorrect configuration propagation.
- Inconsistent logging.
- Potential CUDA synchronization overhead.
- Excessive Python loops.
- CPU/GPU transfers.
- Memory allocation overhead.
- Unnecessary Jacobian computation.
- Incorrect benchmark timing.
- Missing error handling.
- Incomplete experiment metadata.
- Missing scientific validation.

Create a short architecture document:

`docs/ARCHITECTURE.md`

Document the actual architecture, major components, data flow, cache lifecycle, and experiment lifecycle.

Then proceed with implementation.

Do not stop after writing the architecture document.

---

# 4. PHASE 1 — W0 SCIENTIFIC CORRECTNESS AUDIT

This is the highest-priority requirement.

Implement or strengthen the E3 audit infrastructure so that the correctness of Frost's integrated inference path can be established before conducting large experiments.

## 4.1 Required execution modes

The audit should compare the following:

1. Exact MACE baseline.
2. Frost with caching disabled.
3. Frost with all edges dirty.
4. Zeroth-order threshold caching.
5. First-order serial JVP/VJP caching.
6. First-order batched JVP/VJP caching.
7. All-clean diagnostic mode.

All-clean must remain a diagnostic, not a scientifically valid threshold policy.

## 4.2 Correctness checks

Add explicit checks for:

### A. Exact equivalence
- Exact MACE and Frost-disabled must agree.
- Verify forces and relevant intermediate outputs.
- Establish appropriate numerical tolerances.

### B. Cache-disabled equivalence
- Disabled cache must reproduce exact inference.
- No cached approximation may affect outputs.

### C. All-dirty equivalence
- All-dirty Frost must reproduce exact MACE.
- Dirty messages must actually be recomputed.

### D. Message substitution correctness
Verify:
- Dirty edges receive newly computed messages.
- Clean edges receive the intended cached approximation.
- First-order corrections use the correct reference geometry.
- Reference displacement is correct.
- Edge ordering and atom ordering remain consistent.
- Aggregation receives the correct message tensor.

### E. Storage independence
Ensure exact and Frost executions do not accidentally share mutable tensors or cache storage.

Check:
- Cached tensors.
- Reference tensors.
- Intermediate message buffers.
- Output buffers.
- In-place mutation.
- Aliasing.

### F. Zero-displacement control
When the current geometry equals the reference geometry:

\[
\Delta r=0
\]

The first-order approximation should reduce to the reference message.

Verify this explicitly.

Do not assume that zeroth-order and first-order force errors must be identical under all cache policies.

### G. Controlled perturbation
Test controlled perturbations:

- \(10^{-4}\) Å
- \(10^{-3}\) Å
- \(10^{-2}\) Å
- \(5\times10^{-2}\) Å

Measure:
- Message discrepancy.
- Force discrepancy.
- Numerical stability.
- Correct behavior of cache state.

### H. Integrated inference path
Demonstrate that cache substitution affects the intended actual MACE inference path.

Do not validate an isolated approximation while accidentally benchmarking the original exact calculator.

## 4.3 Machine-readable audit

Produce a machine-readable W0 report.

Suggested structure:

`results/active/w0_audit/<audit_id>/`

Include:
- `audit_summary.json`
- `audit_checks.csv`
- `message_errors.csv`
- `force_errors.csv`
- `audit.log`
- `metadata.json`

The summary must contain:
- Test name.
- Pass/fail.
- Numerical tolerance.
- Actual measured discrepancy.
- Relevant configuration.
- Failure reason.
- Software/model versions.
- Execution environment.

The audit must return a nonzero exit code if mandatory correctness checks fail.

Implement an explicit acceptance gate.

Stage B must not execute when W0 fails.

The campaign runner must enforce this gate automatically.

---

# 5. PHASE 2 — MATRIX-FREE JACOBIAN OPTIMIZATION

The current dense Jacobian extraction path is computationally expensive.

The production implementation must avoid materializing dense Jacobians wherever possible.

## 5.1 Production mode: Mode 4 batched JVP/VJP

Use the existing Mode 4 implementation as the preferred first-order cache implementation, provided it passes correctness validation.

The mathematical operation is:

\[
J\Delta r
\]

and any necessary adjoint operation:

\[
J^\top u
\]

The production implementation should use matrix-free operations.

Requirements:

- No dense Jacobian materialization in the production path.
- Avoid unnecessary per-edge Python loops.
- Support batched edge actions.
- Preserve edge ordering.
- Preserve reference geometry consistency.
- Support configurable batch/chunk sizes.
- Avoid unnecessary tensor copies.
- Maintain differentiability where required.
- Ensure numerical equivalence to the dense reference.

## 5.2 Mode 3 serial implementation

Retain Mode 3 as a correctness reference and fallback.

Do not remove it merely because Mode 4 is faster.

Mode 3 and Mode 4 must be compared on identical:
- Configurations.
- Snapshots.
- Tolerances.
- Reference geometries.
- Edge selections.

Compare:
- Message outputs.
- Force outputs.
- RMS error.
- Maximum error.
- Runtime.
- Memory usage.

## 5.3 Dense Jacobian implementation

Retain dense Jacobian extraction only as:
- A small-system correctness oracle.
- A debugging tool.
- A validation reference.

Never make dense Jacobian extraction the default production workflow.

Clearly document its computational cost and intended use.

## 5.4 Configurable batching

Verify that the CLI option:

`--batched-action-chunk-size`

is actually connected to the implementation.

Audit the entire path:

CLI → configuration → experiment runner → cache implementation → batched action evaluator.

Do not allow a CLI option to be silently ignored or replaced by a hardcoded value.

Add tests that confirm different chunk sizes are actually used.

Benchmark chunk sizes such as:

- 64
- 128
- 256
- 512
- 1024

Use the largest practical values without introducing unacceptable memory pressure.

Determine an appropriate default empirically, not by assuming the largest batch is best.

## 5.5 Optional Mode 2 optimization

If the current architecture supports it cleanly, implement or improve chunked `vmap(jacrev)` as an optional experimental mode.

Keep it separate from the primary Mode 4 implementation.

Compare:
- Correctness.
- Runtime.
- Memory consumption.
- CUDA behavior.
- Scalability.

Do not prioritize Mode 2 over completing the primary implementation.

---

# 6. PHASE 3 — CACHE AND GPU PERFORMANCE OPTIMIZATION

Implement all optimizations that are compatible with the current mathematical method.

The following are explicit optimization targets.

## 6.1 Eliminate unnecessary dense operations

Replace dense intermediates with:
- Sparse or indexed operations.
- Direct edge-local evaluation.
- Batched tensor operations.
- Matrix-free action evaluation.

Avoid:
- Dense Jacobian construction.
- Dense edge-by-edge matrices.
- Full message recomputation when only dirty edges need recomputation.
- Unnecessary repeated reference message computation.

## 6.2 Dirty-edge compaction

Investigate and optimize dirty-edge processing.

The intended structure is:

1. Compute the clean/dirty mask.
2. Identify dirty edge indices.
3. Gather dirty edge inputs.
4. Recompute only dirty messages.
5. Scatter or reconstruct the correct message tensor.
6. Apply clean-edge approximations.
7. Continue normal aggregation.

Optimize:
- Index extraction.
- Gather/scatter operations.
- Temporary tensors.
- Python overhead.
- Memory allocations.
- Synchronization.

Preserve the exact edge ordering and numerical behavior.

Benchmark alternative approaches where useful:
- Boolean indexing.
- Explicit index tensors.
- Batched gather/scatter.
- Preallocated buffers.

Choose based on measured performance.

## 6.3 Reuse buffers and minimize allocations

Identify repeated allocations in:
- Message recomputation.
- Cache updates.
- Reference geometry handling.
- JVP/VJP evaluation.
- Message reconstruction.
- Benchmark execution.

Reuse buffers when safe.

Requirements:
- No incorrect aliasing.
- No stale values.
- No unsafe in-place mutation.
- No cross-run contamination.
- Correct behavior across different edge counts.

Document buffer ownership and lifecycle.

## 6.4 GPU execution optimization

Investigate:
- CPU-to-GPU transfers.
- GPU-to-CPU transfers.
- Python loops over edges.
- Frequent CUDA synchronization.
- Unnecessary scalar extraction.
- Repeated kernel launches.
- Repeated allocation.
- Small-batch GPU inefficiency.
- Synchronization inside timing loops.

Do not call `.item()` or synchronize unnecessarily in production code.

Ensure synchronization occurs only when required for correctness or benchmark timing.

Avoid introducing custom CUDA kernels unless there is a compelling measured reason.

## 6.5 Reference cache management

Optimize reference cache operations.

Investigate:
- Reference message storage.
- Reference geometry storage.
- Reference Jacobian/action handling.
- Cache updates.
- Cache age tracking.
- Cache invalidation.
- Reference configuration selection.
- Edge mapping.

Ensure:
- Reference tensors are not accidentally overwritten.
- Cache state is explicit.
- Cache updates are deterministic.
- Reference geometry and reference message correspond to the same configuration.
- Cached actions use the correct reference state.

## 6.6 Avoid unnecessary derivative computation

JVP/VJP should only execute when required by the chosen cache mode and selected edges.

Examples:
- Exact mode: no cache derivative computation.
- Cache disabled: no derivative computation.
- Zeroth-order mode: no first-order derivative computation.
- First-order mode: derivative actions only where required.

Verify this through instrumentation and tests.

## 6.7 Optimization benchmark

Create a benchmark that compares the current and optimized implementations.

Measure:
- Exact message construction.
- Dirty-edge message recomputation.
- Clean-edge reconstruction.
- JVP/VJP.
- Cache bookkeeping.
- Aggregation.
- Full inference.

Report:
- Runtime.
- Memory.
- Speedup.
- Numerical discrepancy.
- Workload size.
- Clean fraction.
- Dirty fraction.

Do not claim optimization based solely on theoretical complexity.

---

# 7. PHASE 4 — STAGE A EXPERIMENT INFRASTRUCTURE

Stage A measures geometric temporal redundancy without Jacobian computation.

Ensure that Stage A remains computationally cheap.

## Requirements

- No Jacobian extraction.
- No derivative computation.
- No force approximation.
- No unnecessary MACE inference.
- Efficient trajectory processing.
- Deterministic sampling.
- Correct periodic geometry.
- Correct edge mapping.

Produce:
- `per_timestep.csv`
- `reuse_summary.csv`
- `trajectory.traj`
- `metadata.json`
- `summary.json`
- `run.log`

Track:
- Timestep.
- Tolerance.
- Clean edges.
- Dirty edges.
- Clean fraction.
- Reference age.
- Edge displacement.
- Edge counts.
- Neighbor-list changes.

Use vectorized geometry calculations where appropriate.

Ensure Stage A outputs can directly drive Stage B without manual intervention.

---

# 8. PHASE 5 — STAGE B ACCURACY AND REUSE FRONTIER

Stage B should be the primary source of reuse–accuracy evidence.

## 8.1 Sampling

Support:
- Early trajectory.
- Middle trajectory.
- Late trajectory.
- Time-stratified sampling.
- Deterministic selection.
- Configurable number of snapshots.

Default research configuration should support 50–100 snapshots per trajectory.

Ensure snapshots are not selected based on observed accuracy.

## 8.2 Tolerance sweep

Support the following tolerances:

- 0.002 Å
- 0.005 Å
- 0.010 Å
- 0.015 Å
- 0.020 Å
- 0.030 Å
- 0.050 Å

Allow configurable tolerances without editing source code.

## 8.3 Cache modes

Support:
- Exact baseline.
- Cache disabled.
- Zeroth-order caching.
- First-order caching.
- Mode 3 serial first-order.
- Mode 4 batched first-order.

Use Mode 4 as the primary production method.

Use Mode 3 as a validation comparison.

## 8.4 Required measurements

At every configuration, snapshot, and tolerance, record:

### Reuse
- Clean edges.
- Dirty edges.
- Total edges.
- Clean fraction.
- Dirty fraction.
- Reference age.
- Reuse count.

### Force error
- RMS force error.
- Maximum force error.
- Mean absolute force error.
- Per-atom force error.
- Per-atom error distribution.
- P95 force error.
- P99 force error.

### Energy error
- Absolute energy error.
- Energy error per atom.
- Relative energy error where meaningful.

### Message error
- Clean-edge message error.
- Dirty-edge message discrepancy.
- Mean message error.
- RMS message error.
- Maximum message error.
- Distribution of message errors.

### Execution
- Cache mode.
- Runtime.
- Configuration.
- Snapshot.
- Tolerance.
- Reference configuration.
- Software versions.

Use stable schemas.

Do not make CSV column names depend on incidental implementation details.

## 8.5 Statistical analysis

Implement:
- Median.
- Mean.
- Standard deviation.
- Interquartile range.
- P90.
- P95.
- P99.
- Maximum.
- Confidence intervals.

Use block bootstrap where temporal correlation exists.

Do not treat correlated snapshots as independent experimental replicates.

Independent trajectory seeds should be treated as a major unit of replication.

## 8.6 Force-error budgets

Support explicit force-error budgets:

- 5 meV/Å.
- 10 meV/Å.
- 20 meV/Å.

Report the largest reuse fraction satisfying each budget.

Do not imply that a tolerance meets a budget merely because its mean error is acceptable.

Report maximum and tail errors as well.

---

# 9. PHASE 6 — EDGE-LEVEL ERROR ANALYSIS

Implement a dedicated analysis module:

`frost/e1/edge_error_analysis.py`

or an appropriately organized analysis package.

This is an important scientific component.

The objective is to understand:

**Does geometric cleanliness actually predict the accuracy of an edge-local cached message?**

## 9.1 Required analysis

Analyze message error as a function of:

- Edge displacement magnitude.
- Reference age.
- Relative edge orientation change.
- Radial distance.
- Distance to cutoff.
- Local environment.
- Edge identity.
- Clean/dirty classification.
- Cache mode.
- Zeroth versus first-order approximation.

Investigate whether edges near the cutoff behave differently.

Investigate whether a simple displacement threshold is a good predictor of message accuracy.

## 9.2 Required outputs

Generate:
- Per-edge message error CSV.
- Per-edge displacement CSV.
- Reference age CSV.
- Grouped statistics.
- Error distributions.
- Clean versus dirty comparisons.
- Zeroth versus first-order comparisons.
- Correlation analysis.
- Regression analysis where appropriate.

## 9.3 Scientific interpretation

Clearly distinguish:
- Correlation.
- Prediction.
- Causal inference.

Do not claim that displacement alone determines error.

This analysis should help identify:
- Which edge characteristics matter.
- Where the threshold policy fails.
- Whether adaptive caching is worth exploring.
- Whether first-order correction helps specific edge classes.

---

# 10. PHASE 7 — REPRODUCIBLE CAMPAIGN RUNNER

Inspect and improve:

`experiments/run_campaign.py`

The campaign runner must provide reliable execution of complete experiments.

## Requirements

- Configuration-driven execution.
- Explicit stage selection.
- Automatic Stage A → W0 → Stage B workflow.
- W0 failure stops Stage B.
- Unique output directories.
- Deterministic seeds.
- Resume support.
- Clear failure reporting.
- Structured logging.
- Metadata generation.
- Configuration validation.
- No silent skipping of experiments.
- No accidental overwriting of results.

Support commands such as:

`--stage all`

`--stage stage-a`

`--stage stage-b`

`--stage audit`

Provide a robust resume mechanism.

If a campaign is interrupted:
- Completed valid stages should remain intact.
- Incomplete stages should be marked.
- Resume should not silently reuse invalid outputs.
- Existing successful results should not be overwritten.

## 10.1 Configuration schema

Improve configuration validation.

Configurations should explicitly specify:
- Experiment name.
- Model.
- Material/system.
- Temperature.
- Seeds.
- Equilibration steps.
- Production steps.
- Sampling protocol.
- Tolerances.
- Cache modes.
- Cache policy.
- Reference update policy.
- JVP/VJP mode.
- Chunk size.
- Output directory.
- Benchmark protocol.

Reject invalid configurations early.

Do not hide defaults that materially affect scientific results.

## 10.2 Metadata

Every experiment must record:
- Git commit.
- Git dirty status.
- Python version.
- PyTorch version.
- CUDA version.
- GPU model.
- Driver version when available.
- MACE version.
- ASE version.
- Configuration.
- Random seed.
- Start/end time.
- Execution status.
- Relevant command-line arguments.

Avoid requiring a Git repository to run small tests, but mark provenance as unavailable when appropriate.

---

# 11. PHASE 8 — GENERALIZATION EXPERIMENTS

Implement or improve the infrastructure for generalization experiments.

## 11.1 Independent trajectory seeds

At minimum support:

- Seed 42.
- Seed 43.
- Seed 44.

Each trajectory should have:
- Independent initialization.
- Independent equilibration.
- Independent production trajectory.
- Its own Stage A.
- Its own W0 audit.
- Its own Stage B.
- Its own statistical summary.

Avoid tuning parameters on every trajectory independently.

Use a predefined calibration/validation protocol.

## 11.2 Temperature generalization

Support:

- 100 K.
- 300 K.
- 600 K.

Use independent seeds.

Report:
- Reuse.
- Force error.
- Stability.
- Structural changes.
- Reference age.
- Edge displacement distributions.

## 11.3 System-size generalization

Support:

- 64 atoms.
- 216 atoms.
- 512 atoms.

Ensure system sizes correspond to valid structures and periodic boundary conditions.

Report:
- Edge count.
- Reuse.
- Runtime.
- Memory.
- Accuracy.
- Scaling behavior.

Do not assume edge counts or runtime scale linearly.

## 11.4 Avoid overengineering

Prioritize:
1. 300 K independent trajectories.
2. Core reuse–accuracy frontier.
3. Inference performance.
4. One meaningful generalization dimension.
5. Longer MD rollout.

Do not prioritize second materials or adaptive policies before core evidence is reliable.

---

# 12. PHASE 9 — W4 INFERENCE PERFORMANCE BENCHMARKING

This phase is critical.

The research must determine whether temporal reuse actually reduces inference cost.

Implement or improve:

`frost/e1/inference_benchmark.py`

and any necessary benchmarking utilities.

## 12.1 Benchmark arms

Support:

1. Exact MACE.
2. Frost wrapper with caching disabled.
3. All-dirty Frost.
4. Controlled 0% clean reuse.
5. Controlled 25% clean reuse.
6. Controlled 50% clean reuse.
7. Controlled 75% clean reuse.
8. Controlled 100% clean reuse.
9. Natural geometric threshold reuse.
10. First-order cache reconstruction.

Controlled clean fractions must be distinct from the natural geometric cache policy.

Do not pretend artificially controlled reuse is a physically realizable policy.

## 12.2 Measurement protocol

Use:
- Fixed configurations.
- Fixed starting states.
- Fixed reference states.
- Warmup evaluations.
- At least 100 timed evaluations when practical.
- Multiple independent repeats.
- CUDA synchronization around complete timed GPU operations.
- Consistent measurement scopes.

Do not include unrelated initialization costs in one arm but not another.

## 12.3 Timing breakdown

Measure:

- Neighbor-list construction.
- Geometry extraction.
- Edge mapping.
- Dirty predicate.
- Dirty-edge compaction.
- Dirty message recomputation.
- JVP/VJP.
- Clean message reconstruction.
- Cache management.
- Aggregation.
- Remaining MACE forward computation.
- Force computation.
- Synchronization overhead.
- Full calculator evaluation.

Ensure component timings do not double-count nested operations.

The complete calculator runtime is the primary performance metric.

Component timings are diagnostic.

## 12.4 Memory

Measure:
- Peak CUDA memory.
- Allocated CUDA memory.
- Reserved CUDA memory.
- CPU memory where practical.
- Cache storage.
- Reference storage.
- Temporary action buffers.

## 12.5 Cost model

Implement a dedicated cost-model analysis module.

Suggested location:

`frost/e1/cost_model_analysis.py`

Use a model conceptually similar to:

\[
T_F =
T_{\mathrm{base}}
-
T_{\mathrm{target}}
+
T_{\mathrm{dirty}}
+
T_{\mathrm{derivative}}
+
T_{\mathrm{geometry}}
+
T_{\mathrm{cache}}
+
T_{\mathrm{sync}}
\]

This is a conceptual decomposition, not an assumption that all components are independent.

Estimate:
- Baseline runtime.
- Targeted computation saved.
- Cache overhead.
- Dirty recomputation overhead.
- Derivative overhead.
- Geometry overhead.
- Synchronization overhead.
- Break-even clean fraction.
- Amdahl-style theoretical upper bound.
- Predicted runtime.
- Actual runtime.
- Prediction residual.

Clearly distinguish:
- Theoretical speedup.
- Predicted speedup.
- Measured speedup.

Do not claim a break-even point unless the measurements support it.

---

# 13. PHASE 10 — W5 END-TO-END MD ROLLOUT

Implement or improve:

`frost/e1/paper_benchmark.py`

The goal is to investigate behavior over multiple MD steps.

## 13.1 Matched starting conditions

Exact and Frost arms must begin from identical:
- Positions.
- Momenta.
- Cell.
- Initial state.
- Thermostat configuration.
- Integrator.
- Timestep.
- Temperature.
- Random state where applicable.

Create a reproducible initial-state artifact.

Store a hash of the initial positions and momenta.

## 13.2 Rollout metrics

Record:

- Runtime per step.
- Cumulative runtime.
- Temperature.
- Kinetic energy.
- Potential energy.
- Total energy.
- Energy drift.
- Cache reuse.
- Dirty fraction.
- Cache age.
- Force errors.
- RMS force discrepancy.
- Maximum force discrepancy.
- P95 force discrepancy.
- Stability/failure indicators.

## 13.3 Exact-force validation

At selected Frost-produced configurations, evaluate exact MACE forces.

Do not compare only the exact and Frost trajectories at identical timestep indices.

MD trajectories may diverge due to chaotic dynamics.

Compare:
- Force accuracy at the same configuration.
- Energy behavior.
- Temperature behavior.
- Structural statistics.
- Stability.
- Runtime.

Do not claim trajectory identity.

## 13.4 Structural validation

Where appropriate, calculate:
- RDF.
- Structural distributions.
- Temperature.
- Energy drift.
- Stability indicators.

Ensure these analyses are computationally reasonable and scientifically appropriate for the system.

---

# 14. PHASE 11 — PAPER ANALYSIS AND FIGURE GENERATION

Create a clean analysis layer.

Prefer:

`analysis/`

Suggested modules:

- `summarize_stage_a.py`
- `summarize_frontier.py`
- `edge_error_analysis.py`
- `summarize_generalization.py`
- `summarize_inference_benchmark.py`
- `cost_model_analysis.py`
- `summarize_rollout.py`
- `generate_paper_figures.py`

Avoid putting research analysis logic directly into low-level model implementation modules.

## 14.1 Required figures

Support automated generation of:

1. Clean-edge fraction versus tolerance.
2. Force error versus tolerance.
3. Reuse–accuracy frontier.
4. Zeroth versus first-order comparison.
5. Error distribution plots.
6. Edge message error versus displacement.
7. Edge message error versus reference age.
8. Generalization across seeds.
9. Temperature generalization.
10. System-size scaling.
11. Inference runtime versus controlled clean fraction.
12. Break-even analysis.
13. MD rollout runtime.
14. MD rollout force-error behavior.

Figures should be:
- Publication quality.
- Consistently formatted.
- Clearly labeled.
- Scientifically interpretable.
- Automatically reproducible from raw results.

Do not hardcode experimental results into figure scripts.

## 14.2 Required summary artifacts

Generate a machine-readable evidence directory:

`results/active/paper_evidence/`

Suggested structure:

- `stage_a_summary.csv`
- `frontier_summary.csv`
- `edge_error_summary.csv`
- `generalization_summary.csv`
- `inference_cost_summary.csv`
- `break_even_summary.csv`
- `rollout_summary.csv`
- `tables/`
- `figures/`

Every artifact must be derived from actual raw experiment outputs.

Missing data must remain missing.

Do not fabricate values or silently replace missing results.

---

# 15. PHASE 12 — SOFTWARE ENGINEERING QUALITY

The repository must become significantly better in quality.

## 15.1 Architecture

Organize code into clear layers:

### Core
- Geometry.
- Cache state.
- Edge mapping.
- Message approximation.
- MACE integration.
- JVP/VJP.
- Cache policy.

### Experiments
- Stage A.
- Stage B.
- W0 audit.
- Generalization.
- Benchmarking.
- Rollout.

### Analysis
- Statistical summaries.
- Frontier analysis.
- Edge analysis.
- Cost modeling.
- Figures.

### Infrastructure
- Configuration.
- Logging.
- Metadata.
- Reproducibility.
- Checkpointing.
- Validation.

Avoid circular dependencies.

Keep model implementation separate from experiment orchestration.

## 15.2 Code standards

Use:
- Type hints.
- Docstrings for public APIs.
- Clear variable names.
- Small functions with explicit responsibilities.
- Explicit data structures where appropriate.
- Consistent exception handling.
- Structured logging.
- Centralized configuration.
- Consistent naming.
- Clear numerical tolerances.

Avoid:
- Giant monolithic functions.
- Excessive nested conditions.
- Unnecessary abstraction.
- Global mutable state.
- Silent exception handling.
- Hardcoded paths.
- Hardcoded experimental parameters.
- Hidden fallback behavior.
- Unexplained magic numbers.

## 15.3 Configuration

Use validated configuration structures.

If the existing project uses JSON, preserve compatibility where reasonable.

Introduce schema validation using an appropriate dependency only if justified.

Avoid excessive dependencies.

## 15.4 Logging

Use consistent logging instead of scattered prints.

Support:
- INFO.
- DEBUG.
- WARNING.
- ERROR.

Log:
- Experiment lifecycle.
- Stage completion.
- Important configuration.
- Cache behavior.
- Timing.
- Failures.
- Audit results.

Avoid verbose per-edge logging in production experiments.

## 15.5 Error handling

Handle:
- Invalid configurations.
- Missing files.
- Corrupt outputs.
- GPU unavailable.
- CUDA OOM.
- Interrupted experiments.
- Incorrect edge mapping.
- MACE API incompatibility.

Fail loudly for scientific correctness failures.

Never silently substitute exact inference when Frost is supposed to be running.

If a fallback is unavoidable, record it explicitly and invalidate the affected performance result.

## 15.6 Testing

Improve the test suite substantially.

Include:

### Unit tests
- Geometry calculations.
- Edge mapping.
- Cache state.
- Dirty/clean classification.
- Reference updates.
- Message reconstruction.
- JVP/VJP.
- Batch handling.
- Chunk size propagation.
- Cache invalidation.

### Numerical tests
- Exact equivalence.
- Dense versus matrix-free actions.
- Mode 3 versus Mode 4.
- Zero-displacement control.
- First-order correction.
- Perturbation behavior.

### Integration tests
- Exact MACE.
- Cache disabled.
- All-dirty.
- First-order cache.
- Actual message substitution.
- Aggregation.
- Force output.

### Experiment tests
- Stage A smoke test.
- Stage B smoke test.
- W0 audit.
- Campaign execution.
- Resume behavior.
- Metadata.
- Output schemas.

### Benchmark tests
- Correct timing scope.
- Correct benchmark arms.
- No accidental fallback.
- Synchronization.
- Result integrity.

Use small systems and short trajectories for tests.

Do not run expensive production campaigns as ordinary unit tests.

Separate:
- Fast unit tests.
- Integration tests.
- GPU tests.
- Slow research tests.

## 15.7 Test execution

Run:
- Formatting.
- Linting.
- Type checking where feasible.
- Unit tests.
- Integration tests.
- GPU tests if the environment supports them.

Use the existing project conventions if already established.

Report all failures.

Do not claim tests passed unless they were actually executed.

---

# 16. PHASE 13 — DOCUMENTATION

Update or create the following:

## `README.md`
Include:
- Project objective.
- Research questions.
- Installation.
- Environment setup.
- Repository structure.
- Experiment overview.
- Quick-start smoke test.
- Campaign commands.
- Benchmark commands.
- Result locations.

## `docs/ARCHITECTURE.md`
Actual software architecture.

## `docs/EXPERIMENT_ROADMAP.md`
Updated research execution roadmap.

## `docs/EXPERIMENT_PROTOCOL.md`
Detailed scientific protocols:
- Stage A.
- W0.
- Stage B.
- Generalization.
- W4.
- W5.
- Statistical methodology.
- Error budgets.
- Benchmark methodology.

## `docs/PERFORMANCE.md`
Performance methodology:
- Timing scopes.
- GPU synchronization.
- Benchmark arms.
- Cost model.
- Memory measurement.
- Interpretation of speedup.

## `docs/RESULTS_SCHEMA.md`
Document all CSV/JSON output schemas.

## `docs/REPRODUCIBILITY.md`
Explain:
- Seeds.
- Metadata.
- Environment capture.
- Git commit.
- Configuration.
- Deterministic execution.
- Output organization.
- Reproduction of results.

## `docs/OPTIMIZATION_LOG.md`
Record:
- Optimization.
- Motivation.
- Implementation.
- Correctness validation.
- Performance measurement.
- Tradeoffs.
- Whether it was retained.

Do not report unmeasured optimizations as successful.

---

# 17. PHASE 14 — CONFIGURATION FILES

Inspect and improve the existing configurations.

Maintain appropriate configuration files for:

- Pilot.
- Generalization.
- Temperature.
- Scale.
- Inference benchmark.
- Rollout.

Avoid duplicating configuration structures unnecessarily.

Ensure each config has:
- Valid schema.
- Explicit experimental parameters.
- Appropriate output directory.
- Clear seed configuration.
- Clear cache mode.
- Clear tolerances.
- Clear sampling protocol.

Create a small smoke-test configuration.

Smoke tests must never be confused with publication experiments.

---

# 18. REQUIRED RESEARCH EXECUTION ORDER

The software should support the following execution order.

## P0 — Pilot correctness

1. Stage A smoke test.
2. W0 audit.
3. Stage B pilot.
4. Mode 3 versus Mode 4 validation.
5. Chunk-size validation.

Do not proceed if W0 fails.

## P1 — Core reuse–accuracy evidence

1. Stage A.
2. 50–100 snapshots.
3. Seven tolerances.
4. Zeroth-order and first-order.
5. Complete force-error metrics.
6. Edge-level message analysis.
7. Statistical summaries.
8. Frontier figures.

## P1 — Independent trajectory generalization

1. Seeds 42, 43, 44.
2. Independent Stage A.
3. Independent W0.
4. Stage B.
5. Generalization summaries.

## P1 — Performance

1. Exact baseline.
2. Frost-disabled.
3. All-dirty.
4. Controlled clean fractions.
5. Natural threshold policy.
6. First-order cache.
7. Timing breakdown.
8. Cost model.
9. Break-even analysis.

## P2 — Generalization

Run one of:
- Temperature.
- System size.

Prioritize the one that offers the strongest scientific value.

## P2 — MD rollout

Run short pilot first.

Only then run longer rollout experiments.

## P3 — Optional

- Adaptive cache policies.
- Second material.
- Additional model.
- More advanced performance optimization.

Do not make optional work block the core paper.

---

# 19. IMPLEMENTATION EXECUTION PLAN

You must work incrementally.

## Step 1 — Audit

Inspect the repository and produce:

`docs/IMPLEMENTATION_AUDIT.md`

Include:
- Existing features.
- Missing features.
- Incorrect implementations.
- Optimization opportunities.
- Relevant existing tests.
- Proposed changes.
- File-by-file implementation plan.

Then continue implementation.

## Step 2 — Establish baseline

Before changing performance-critical code:

- Run relevant existing tests.
- Run a small baseline experiment.
- Record current behavior.
- Record baseline performance if feasible.
- Preserve the baseline outputs.

If expensive experiments cannot be run, explain why and use a smaller valid smoke test.

## Step 3 — Implement correctness infrastructure

Implement W0 improvements first.

Run correctness tests.

Do not proceed with scientific campaign automation until correctness gating is functional.

## Step 4 — Implement matrix-free optimization

Implement Mode 4 improvements.

Validate:
- Numerical equivalence.
- Chunk sizes.
- Edge ordering.
- Integrated message substitution.

Benchmark against Mode 3 and dense references.

## Step 5 — Implement cache/GPU optimizations

Proceed with dirty-edge compaction, buffer reuse, allocation reduction, and synchronization improvements.

For every optimization:
- Establish baseline.
- Implement.
- Test.
- Benchmark.
- Record results.

Do not combine many unvalidated changes into one opaque patch.

## Step 6 — Improve experimental infrastructure

Implement:
- Stage A.
- Stage B.
- Campaign runner.
- Resume.
- Metadata.
- Output validation.
- Statistical analysis.

## Step 7 — Implement research analysis

Implement:
- Edge error analysis.
- Frontier analysis.
- Generalization summaries.
- Cost modeling.
- Figure generation.

## Step 8 — Implement W4 and W5

Build rigorous inference benchmarking and rollout infrastructure.

Ensure correct measurement scope.

## Step 9 — Final validation

Run all feasible tests.

Run smoke experiments.

Inspect generated outputs.

Verify:
- CSV schemas.
- JSON schemas.
- Metadata.
- Reproducibility.
- No accidental overwrites.
- Correct output paths.
- No silent fallbacks.

## Step 10 — Documentation and final report

Update documentation.

Produce:

`docs/IMPLEMENTATION_REPORT.md`

---

# 20. FINAL IMPLEMENTATION REPORT

At completion, provide a detailed report containing:

## A. Architecture
- What the codebase now looks like.
- Major modules.
- Data flow.
- Cache lifecycle.
- Experiment lifecycle.

## B. Implemented features
List every major feature implemented.

## C. Optimizations
For each optimization:
- Previous implementation.
- New implementation.
- Motivation.
- Correctness validation.
- Measured performance change.
- Memory change.
- Limitations.

## D. Scientific correctness
- W0 results.
- Numerical comparisons.
- Dense versus matrix-free validation.
- Mode 3 versus Mode 4.
- Known limitations.

## E. Testing
- Tests executed.
- Tests passed.
- Tests failed.
- Tests not run.
- Reason for failures.

## F. Experiment infrastructure
- Stage A.
- Stage B.
- Campaign runner.
- Generalization.
- W4.
- W5.
- Analysis pipeline.

## G. Research readiness
Identify:
- Which experiments are ready to execute.
- Which results are currently available.
- Which scientific questions remain unanswered.
- Which features are optional.
- Which limitations still exist.

## H. Exact next commands
Provide the exact commands to:
1. Run smoke tests.
2. Run W0.
3. Run pilot Stage B.
4. Run the core frontier.
5. Run generalization.
6. Run W4.
7. Run W5.
8. Generate paper figures and tables.

---

# 21. IMPLEMENTATION PRIORITIES AND CONSTRAINTS

The priorities are:

1. Scientific correctness.
2. Numerical reliability.
3. Matrix-free computational efficiency.
4. Efficient cache implementation.
5. Reproducibility.
6. Experimental validity.
7. Performance measurement.
8. Maintainability.
9. Publication-quality analysis.
10. Optional extensions.

Do not sacrifice correctness for speed.

Do not sacrifice experimental validity for convenience.

Do not introduce a framework merely because it is fashionable.

Prefer simple, well-tested, maintainable implementations.

Use established scientific computing practices and appropriate numerical methods.

Avoid unnecessary dependencies.

Do not make large refactors without a clear justification.

Keep the repository runnable on the current GPU environment.

Preserve compatibility with the existing MACE/ASE stack unless a version change is necessary and validated.

---

# 22. YOUR IMMEDIATE TASK

Start now.

1. Inspect the complete repository.
2. Read the current README and roadmap.
3. Read the existing implementation and tests.
4. Produce `docs/IMPLEMENTATION_AUDIT.md`.
5. Identify the highest-priority gaps.
6. Establish the baseline.
7. Begin implementing the improvements in the priority order above.

Do not stop after the audit or a proposed plan.

Implement the changes.

If the task is too large for one uninterrupted execution:
- Complete a coherent implementation phase.
- Commit no incomplete scientific behavior.
- Update the implementation report.
- Clearly identify remaining work.
- Continue with the next phase when possible.

If you encounter a technical obstacle:
- Investigate it.
- Implement the best validated alternative.
- Document the limitation.
- Continue with independent tasks.

Do not fabricate successful tests or performance improvements.

The final objective is a **scientifically rigorous, computationally efficient, reproducible, maintainable Frost research codebase**, capable of supporting a serious research paper on temporal edge reuse in geometric GNN inference.

Begin with repository inspection and then execute the implementation plan.
