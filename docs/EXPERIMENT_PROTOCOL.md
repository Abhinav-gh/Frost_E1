# Experiment protocol

1. Install a CUDA-compatible PyTorch and MACE build on the GPU VM.
2. Run the smoke/pilot campaign in a fresh output directory.
3. Inspect W0 `summary.json`; do not run or interpret Stage B if
   `audit_passed` is false.
4. Run Stage A once per trajectory and retain its trajectory and metadata.
5. Run Stage B with time-stratified snapshots and the declared tolerance list.
6. Treat seeds/trajectories as independent replicates; snapshots within one
   trajectory are temporally correlated.
7. Run W4 with warmups, synchronized complete evaluations, and repeated arms.
8. Run W5 only from matched post-equilibration positions and momenta.

The all-clean and controlled-fraction W4 arms are diagnostics and must not be
described as physically realizable threshold policies. Missing data remains
missing. No tolerance may be selected after inspecting force error.
