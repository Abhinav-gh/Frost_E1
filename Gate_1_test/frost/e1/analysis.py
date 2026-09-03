"""
Frost E1 — analysis.py

Post-run analysis of E1 HDF5 logs.

Computes:
  1. Dirty fraction vs tolerance curves
  2. Spatial autocorrelation of dirty masks
  3. Cache lifetime distributions
  4. Force-error vs perturbation joint statistics
  5. Summary tables for the E1 go/no-go gate
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_trajectory_stats(hdf5_path: Path) -> Tuple[dict, pd.DataFrame, dict]:
    """
    Load E1 HDF5 log into metadata + per-step DataFrame + force-error data.

    Returns
    -------
    metadata : dict
    steps_df : pd.DataFrame  (one row per production step)
    force_errors : dict {epsilon: list of per-frame stats}
    """
    with h5py.File(hdf5_path, "r") as f:
        # Load metadata
        metadata = dict(f["metadata"].attrs)

        # Load per-step data
        steps = f["steps"]
        n_steps = steps["timestep"].shape[0]
        if n_steps == 0:
            return metadata, pd.DataFrame(), {}

        data = {"timestep": steps["timestep"][:]}
        for key in steps.keys():
            if key != "timestep":
                data[key] = steps[key][:]

        # Load tolerance data
        tol_grp = f["tolerances"]
        for k in tol_grp.keys():
            eps = tol_grp[k].attrs["epsilon_meV_per_A"]
            data[f"dirty_fraction_eps_{eps:.4f}"] = tol_grp[k]["dirty_fraction"][:]
            data[f"dirty_count_eps_{eps:.4f}"] = tol_grp[k]["dirty_count"][:]

        steps_df = pd.DataFrame(data)

        # Load force error frames
        force_errors = {}
        if "force_error_frames" in f:
            for frame_key in f["force_error_frames"].keys():
                frm = f["force_error_frames"][frame_key]
                eps = frm.attrs["epsilon"]
                if eps not in force_errors:
                    force_errors[eps] = []
                force_errors[eps].append({
                    "timestep": frm.attrs["timestep"],
                    "epsilon": eps,
                    "max_error": frm.attrs["max_error"],
                    "mean_error": frm.attrs["mean_error"],
                    "rms_error": frm.attrs["rms_error"],
                    "p50_error": frm.attrs.get("p50_error", np.nan),
                    "p90_error": frm.attrs.get("p90_error", np.nan),
                    "p99_error": frm.attrs.get("p99_error", np.nan),
                })

    return metadata, steps_df, force_errors


def compute_dirty_fraction_summary(steps_df: pd.DataFrame, tolerances: List[float]) -> pd.DataFrame:
    """
    For each tolerance, compute mean/median/IQR of dirty fraction over all production steps.
    """
    rows = []
    for eps in tolerances:
        col = f"dirty_fraction_eps_{eps:.4f}"
        if col not in steps_df.columns:
            continue
        vals = steps_df[col].values
        rows.append({
            "epsilon": eps,
            "mean_dirty_fraction": float(np.mean(vals)),
            "median_dirty_fraction": float(np.median(vals)),
            "std_dirty_fraction": float(np.std(vals)),
            "p25_dirty_fraction": float(np.percentile(vals, 25)),
            "p75_dirty_fraction": float(np.percentile(vals, 75)),
            "iqr_dirty_fraction": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "min_dirty_fraction": float(np.min(vals)),
            "max_dirty_fraction": float(np.max(vals)),
            "n_steps": len(vals),
        })
    return pd.DataFrame(rows)


def compute_cache_lifetime_from_dirty_fractions(
    steps_df: pd.DataFrame,
    tolerances: List[float],
) -> dict:
    """
    Estimate cache lifetime distribution from dirty fraction time series.

    For each tolerance, estimate per-edge lifetime from the dirty fraction
    time series. An exact per-edge lifetime requires raw per-edge data,
    which we don't store for the full trajectory. This is an approximation
    using the run-length distribution of the "clean" state.
    """
    results = {}
    for eps in tolerances:
        col = f"dirty_fraction_eps_{eps:.4f}"
        if col not in steps_df.columns:
            continue
        dirty_frac = steps_df[col].values
        # Clean fraction per step
        clean_frac = 1.0 - dirty_frac
        # Mean clean fraction -> estimate mean lifetime = 1 / mean_dirty_prob
        mean_dirty = np.mean(dirty_frac)
        if mean_dirty > 0:
            estimated_mean_lifetime = 1.0 / mean_dirty
        else:
            estimated_mean_lifetime = len(dirty_frac)  # never dirty
        results[eps] = {
            "epsilon": eps,
            "estimated_mean_lifetime_steps": float(estimated_mean_lifetime),
            "mean_dirty_fraction": float(mean_dirty),
            "median_dirty_fraction": float(np.median(dirty_frac)),
        }
    return results


def compute_morans_I(
    dirty_mask: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float = 10.0,
) -> float:
    """
    Compute Moran's I spatial autocorrelation of the dirty atom mask.

    Moran's I measures whether dirty atoms cluster spatially (I > 0)
    or are dispersed (I < 0). I ≈ 0 means random spatial distribution.

    Definition:
        I = (N / W) * (sum_{i,j} w_{ij} * z_i * z_j) / (sum_i z_i^2)

    where:
        z_i = dirty_mask[i] - mean(dirty_mask)
        w_{ij} = 1 if atoms i,j are within cutoff, 0 otherwise
        W = sum_{i,j} w_{ij}
        N = number of atoms

    Parameters
    ----------
    dirty_mask : np.ndarray, shape (N,), dtype bool
        True for dirty atoms.
    positions : np.ndarray, shape (N, 3)
        Atom positions.
    cell : np.ndarray, shape (3, 3)
        Simulation cell.
    cutoff : float
        Spatial cutoff for spatial weight matrix (Angstrom).

    Returns
    -------
    I : float
        Moran's I statistic.
    """
    from ase.neighborlist import primitive_neighbor_list
    from ase import Atoms

    n_atoms = len(positions)
    if n_atoms < 2:
        return 0.0

    z = dirty_mask.astype(float) - np.mean(dirty_mask.astype(float))

    # Build spatial weight matrix using neighbor list
    atoms_temp = Atoms(
        symbols=["H"] * n_atoms,
        positions=positions,
        cell=cell,
        pbc=True,
    )
    i_idx, j_idx, _ = primitive_neighbor_list(
        "ijS",
        pbc=atoms_temp.get_pbc(),
        cell=atoms_temp.get_cell().array,
        positions=atoms_temp.get_positions(),
        cutoff=cutoff,
        self_interaction=False,
    )

    W = len(i_idx)  # total number of neighbor pairs
    if W == 0:
        return 0.0

    numerator = np.sum(z[i_idx] * z[j_idx])
    denominator = np.sum(z ** 2)

    if denominator == 0:
        return 0.0  # all atoms same state -> no spatial pattern

    I = (n_atoms / W) * (numerator / denominator)
    return float(I)


def apply_gonogo_criterion(
    dirty_summary_df: pd.DataFrame,
    force_error_summary: dict,
    favorable_systems: List[str] = ("Si", "SiO2"),
    go_nogo_force_threshold: float = 5.0,  # meV/Angstrom
    go_nogo_dirty_threshold: float = 0.5,
) -> dict:
    """
    Apply the Frost go/no-go criterion from the research document.

    Criterion (from document):
    If phi > 0.5 at the largest epsilon whose induced force error
    remains below 5 meV/A on favorable systems (Si, silica glass),
    then the mechanism does not deliver.

    Parameters
    ----------
    dirty_summary_df : pd.DataFrame
        Output from compute_dirty_fraction_summary.
    force_error_summary : dict
        {epsilon: {'max_error', 'mean_error', 'rms_error'}} from force error experiment.
    favorable_systems : list
        Systems that are predicted to be most favorable (from paper).
    go_nogo_force_threshold : float
        Maximum acceptable force error (meV/Angstrom). From paper: 5 meV/A.
    go_nogo_dirty_threshold : float
        Maximum acceptable dirty fraction at the threshold epsilon. From paper: 0.5.

    Returns
    -------
    verdict : dict with keys:
        'pass': bool
        'explanation': str
        'largest_valid_epsilon': float or None
        'dirty_fraction_at_gate': float or None
        'force_error_at_gate': float or None
        'gate_force_threshold': float
        'gate_dirty_threshold': float
    """
    # Find the largest epsilon where mean force error < 5 meV/A
    # Convert 5 meV/A to eV/A for comparison if needed
    # Note: force_error_summary uses eV/A units (from MACE output)
    # 5 meV/A = 0.005 eV/A
    force_threshold_eV = go_nogo_force_threshold * 1e-3  # convert meV/A to eV/A

    valid_epsilons = []
    for eps, fe_stats in sorted(force_error_summary.items()):
        # Use mean_error for the gate comparison
        if fe_stats.get("mean_error", float("inf")) < force_threshold_eV:
            valid_epsilons.append(eps)

    if not valid_epsilons:
        return {
            "pass": False,
            "explanation": (
                "No epsilon value produced mean force error < "
                f"{go_nogo_force_threshold} meV/A. "
                "All tolerances exceed the force accuracy budget."
            ),
            "largest_valid_epsilon": None,
            "dirty_fraction_at_gate": None,
            "force_error_at_gate": None,
            "gate_force_threshold": go_nogo_force_threshold,
            "gate_dirty_threshold": go_nogo_dirty_threshold,
        }

    largest_valid_eps = max(valid_epsilons)

    # Get dirty fraction at that epsilon
    eps_row = dirty_summary_df[
        np.isclose(dirty_summary_df["epsilon"], largest_valid_eps)
    ]
    if eps_row.empty:
        dirty_frac = None
    else:
        dirty_frac = float(eps_row["median_dirty_fraction"].iloc[0])

    force_error_at_gate = force_error_summary[largest_valid_eps].get("mean_error", None)

    if dirty_frac is None:
        passes = False
        explanation = "Could not determine dirty fraction at gate epsilon."
    elif dirty_frac > go_nogo_dirty_threshold:
        passes = False
        explanation = (
            f"FAIL: At largest valid epsilon ({largest_valid_eps:.3f} meV/A), "
            f"dirty fraction phi={dirty_frac:.3f} > threshold {go_nogo_dirty_threshold:.1f}. "
            "The mechanism does not deliver sufficient reuse."
        )
    else:
        passes = True
        explanation = (
            f"PASS: At largest valid epsilon ({largest_valid_eps:.3f} meV/A), "
            f"dirty fraction phi={dirty_frac:.3f} < threshold {go_nogo_dirty_threshold:.1f}. "
            "Sufficient temporal redundancy exists to justify further development."
        )

    return {
        "pass": passes,
        "explanation": explanation,
        "largest_valid_epsilon": largest_valid_eps,
        "dirty_fraction_at_gate": dirty_frac,
        "force_error_at_gate": force_error_at_gate,
        "gate_force_threshold": go_nogo_force_threshold,
        "gate_dirty_threshold": go_nogo_dirty_threshold,
    }
