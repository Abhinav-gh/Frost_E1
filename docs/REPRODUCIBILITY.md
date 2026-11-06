# Reproducibility

Record the Git commit, dirty-worktree state, Python/PyTorch/CUDA versions,
model name, device, dtype, seed, configuration, initial-state hash, and output
paths. Use a fresh output directory for every run. Do not overwrite legacy
results.

The campaign runner uses deterministic seed/configuration arguments, saves
Stage-A trajectories, and refuses non-empty stage directories. A resumed
campaign must point at the same configuration and existing valid outputs;
incomplete output directories should be moved aside rather than silently
reused.

GPU timing requires synchronization. Numerical results from one machine,
model checkpoint, or dtype should not be generalized to another without a
separate campaign.
