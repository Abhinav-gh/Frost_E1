# Frost E1 — Usage Guide

This document explains how to run the E1 Redundancy Characterisation codebase for the Frost project.

## 1. Environment Setup

The E1 experiment relies on a specific conda environment containing PyTorch (with CUDA 12.4), ASE, MACE-Torch, and standard scientific Python packages.

*(If you haven't built the environment yet, the script installed PyTorch 2.4.0+cu124 and `mace-torch`, along with `ase`, `h5py`, `pandas`, `matplotlib` etc.)*

## 2. Running Unit Tests

We have written unit tests to verify the core geometric correctness (periodic boundary conditions, minimum image convention) and the per-edge simulated cache logic.

To run the test suite, ensure your `PYTHONPATH` includes the current directory and activate the environment:
```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate frost_e1
PYTHONPATH=. pytest tests/
```
You should see 8 passing tests.

## 3. Running the Pilot Experiment

The pilot experiment is a shortened run that executes exactly 1 trajectory out of the 60 in the production matrix. It uses the `[pilot]` configuration section in `configs/e1.yaml` (MACE-MP-0 on Silicon at 300 K for 100 steps).

This validates the entire pipeline: system generation, MD driver, edge caching, force-error sampling, HDF5 logging, and plot generation.

Run the pilot using:
```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate frost_e1
PYTHONPATH=. python -m frost.e1 --config configs/e1.yaml --pilot
```

## 4. Running the Full Production Matrix

The full matrix evaluates **MACE-MP-0** across 5 benchmark systems (Si, Cu, Li3PO4, Liquid Water, Silica Glass) at 2 temperatures (300K, 1000K) with 3 random seeds. This generates a total of 30 production trajectories (60 once an Allegro checkpoint becomes available).

**NOTE:** The complete matrix involves roughly 300,000 steps of MACE-MP-0 evaluation and will take a significant amount of time on a single RTX A6000. It is recommended to run this inside a `tmux` session or background it using `nohup`.

To run the full suite:
```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate frost_e1
PYTHONPATH=. python -m frost.e1 --config configs/e1.yaml
```

## 5. Understanding the Outputs

All generated files are saved into the `results/e1/` directory:

- `results/e1/raw/*.h5`
  Streaming HDF5 files per trajectory containing per-step aggregated metrics (mean/median/std $\Delta r$ and $\Delta\hat{r}$, dirty fractions per tolerance) and sampled exact/stale force vectors.

- `results/e1/plots/*.png`
  The 6 required figures characterizing the temporal redundancy of the systems:
  1. `dirty_fraction_vs_tolerance.png`
  2. `perturbation_force_error_joint.png`
  3. `delta_r_vs_delta_rhat.png`
  4. `force_error_by_tolerance.png`
  5. `cache_lifetime.png`
  6. `spatial_autocorrelation.png`

- `docs/e1_results.md`
  A programmatic summary outputted at the end of the script indicating a definitive **PASS/FAIL** verdict against the Frost Phase-1 Gate Criterion ($\phi < 0.5$ at $5$ meV/Å tolerance).
