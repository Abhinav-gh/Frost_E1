# Frost E1 — Environment Inspection Report

**Date:** 2026-09-02  
**Purpose:** Pre-implementation environment audit for Gate 1 — E1 Redundancy Characterisation

---

## 1. Hardware

| Component | Detail |
|-----------|--------|
| **GPU** | 1× NVIDIA RTX A6000 (single GPU; paper specifies 2× A6000) |
| **VRAM** | 48,539 MiB (~47.4 GB) free at inspection |
| **GPU Driver** | 575.57.08 |
| **CUDA Runtime (driver)** | 12.9 |
| **CUDA Toolkit** | 12.4 (active, matches PyTorch build) |
| **CPU** | AMD EPYC 7542 32-Core Processor (10 vCPUs visible) |
| **RAM** | ~32 GB total, ~29 GB available |
| **Disk** | 1.5 TB total, ~95 GB free (94% used) |

WARNING: Only 1 GPU present (paper assumes 2x RTX A6000). All experiments run sequentially.
WARNING: Disk at 94% — storage strategy must be conservative (aggregated HDF5, not raw CSV).

---

## 2. CUDA Toolkit Installations

  /usr/local/cuda-11.3, cuda-11.7, cuda-11.8, cuda-12, cuda-12.4
  Active: /usr/local/cuda -> cuda-12.4
  PyTorch build: 2.4.0+cu124 — PERFECT MATCH

---

## 3. Python Environments Found

| Environment | Python | PyTorch | MLIP stack |
|-------------|--------|---------|------------|
| base (conda /opt/conda) | 3.11.5 | NOT installed | NOT installed |
| adapter_env | 3.10.14 | 2.4.0+cu124 | NOT installed |
| adapter_env_packed | 3.10.14 | 2.4.0+cu124 | NOT installed |
| **frost_e1 (to create)** | **3.10** | **2.4.0+cu124** | **to install** |

Decision: New isolated environment `frost_e1` will be created.

---

## 4. Software in adapter_env (reference)

| Package | Version |
|---------|---------|
| Python | 3.10.14 |
| PyTorch | 2.4.0+cu124 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| pandas | 2.3.3 |
| matplotlib | 3.10.9 |
| tqdm | 4.67.3 |
| PyYAML | 6.0.3 |

---

## 5. Missing MLIP Dependencies (Critical)

| Package | Status | Resolution |
|---------|--------|-----------|
| ASE | MISSING | pip install ase |
| e3nn | MISSING | pip install e3nn |
| mace-torch | MISSING | pip install mace-torch |
| nequip | MISSING | pip install nequip |
| allegro (GitHub) | MISSING | see §8 |
| h5py | MISSING | pip install h5py |
| pymatgen | MISSING | pip install pymatgen (optional) |

---

## 6. Pretrained Model Checkpoints

Search result: NONE FOUND on local filesystem.

| Model | Source | Status |
|-------|--------|--------|
| MACE-MP-0 (medium) | HuggingFace ACEsuit/mace-mp | NOT present; auto-downloads on first use |
| Allegro (universal) | No public universal checkpoint exists | BLOCKER — see §8 |

---

## 7. Benchmark System Structures

Search result: NONE FOUND locally. All will be constructed programmatically:

| System | N_atoms | Construction |
|--------|---------|-------------|
| Diamond Si | 64 | ase.build.bulk('Si','diamond',a=5.43) x 2x2x2 |
| Cu FCC | 108 | ase.build.bulk('Cu','fcc',a=3.615) x 3x3x3 |
| Li3PO4 | ~112 | gamma-phase crystal, Pnma, embedded CIF or pymatgen |
| Liquid water | 192 | Pre-equilibrated 64xH2O box, standard config |
| Silica glass | 216 | Melt-quench MD protocol |

---

## 8. Allegro Blocker (CRITICAL)

Allegro has NO publicly available universal pretrained checkpoint
analogous to MACE-MP-0. The Frost paper likely uses custom-trained
models per system.

Resolution for E1:
  - Start with MACE-MP-0 (universal, covers all 5 systems)
  - Attempt Allegro installation for structural smoke-test only
  - Document explicitly that Allegro results require trained checkpoints
  - Do NOT fabricate checkpoint data

---

## 9. Storage Estimates

Per trajectory (aggregated stats, not raw per-edge):
  ~50–100 MB per trajectory

Full 60-trajectory matrix:
  ~3–6 GB (safe given 95 GB free)

Per-trajectory raw per-edge storage (NOT recommended):
  ~0.3–0.7 GB per trajectory → ~18–42 GB total → borderline

Decision: Aggregate per-step statistics to HDF5.
Store raw per-edge data only for 100 force-error sample frames.

---

## 10. GPU Memory Estimates

| Model | System | Estimated VRAM |
|-------|--------|---------------|
| MACE-MP-0 medium | 64-atom Si | ~2-4 GB |
| MACE-MP-0 medium | 192-atom water | ~3-6 GB |
| MACE-MP-0 medium | 216-atom SiO2 | ~4-8 GB |

A6000 48 GB VRAM is ample. No OOM risk expected.

---

## 11. Network Access

PyPI: REACHABLE
HuggingFace: REACHABLE

---

## 12. Summary of Blockers

| Blocker | Severity | Resolution |
|---------|---------|-----------|
| No ASE | CRITICAL | Install in frost_e1 env |
| No MACE-MP-0 checkpoint | MEDIUM | Auto-downloads on first use |
| No Allegro pretrained checkpoint | CRITICAL | Start with MACE-MP-0 only |
| Disk at 94% | MEDIUM | Aggregated HDF5 storage |
| Single GPU vs paper 2x | LOW | Sequential runs acceptable |
