"""
Frost E1 — geometry.py

Edge geometry primitives for the E1 Redundancy Characterisation experiment.

IMPORTANT: All displacement vectors use the minimum-image convention (MIC)
with respect to the simulation cell. Do NOT use naive r_j - r_i; that
ignores periodic boundary conditions and produces wrong edge geometry.

The edge geometry consumed by MACE/Allegro is:
    r_ij  = r_j - r_i  (MIC-corrected displacement vector, shape [3])
    d_ij  = ||r_ij||   (scalar distance)
    rhat_ij = r_ij / d_ij  (unit direction vector, shape [3])

Edge identity: (atom_i, atom_j, shift) where shift is the periodic
image offset vector (integer, shape [3]).

The shift is essential: two atoms can have multiple edges connecting them
via different periodic images. The MLIP cutoff determines which images
contribute.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple


def minimum_image_displacement(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    cell: np.ndarray,
    shift: np.ndarray,
) -> np.ndarray:
    """
    Compute the minimum-image displacement vector r_ij = r_j - r_i + shift @ cell.

    Parameters
    ----------
    pos_i : np.ndarray, shape (3,)
        Position of atom i in Cartesian coordinates.
    pos_j : np.ndarray, shape (3,)
        Position of atom j in Cartesian coordinates.
    cell : np.ndarray, shape (3, 3)
        Simulation cell matrix where rows are lattice vectors.
        cell[0] = a-vector, cell[1] = b-vector, cell[2] = c-vector.
    shift : np.ndarray, shape (3,), dtype int
        Periodic image shift vector (integer). The actual displacement
        correction is shift @ cell.

    Returns
    -------
    r_ij : np.ndarray, shape (3,)
        Displacement vector from atom i to atom j, MIC-corrected.
    """
    return pos_j - pos_i + shift @ cell


def compute_edge_geometry(
    r_ij: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Compute scalar distance and unit direction from displacement vector.

    Parameters
    ----------
    r_ij : np.ndarray, shape (3,)
        MIC-corrected displacement vector.

    Returns
    -------
    d_ij : float
        Scalar distance ||r_ij||.
    rhat_ij : np.ndarray, shape (3,)
        Unit direction vector r_ij / ||r_ij||.

    Raises
    ------
    ValueError
        If ||r_ij|| == 0 (degenerate edge, should not occur in valid NL).
    """
    d_ij = float(np.linalg.norm(r_ij))
    if d_ij == 0.0:
        raise ValueError(
            f"Zero-length edge displacement: r_ij={r_ij}. "
            "This indicates a self-edge or degenerate periodic image."
        )
    rhat_ij = r_ij / d_ij
    return d_ij, rhat_ij


def build_edge_geometry_from_ase(
    atoms,
    cutoff: float,
) -> dict:
    """
    Build the complete edge geometry table from an ASE Atoms object.

    Uses ASE's neighbour list with minimum-image convention.
    Returns a dictionary keyed by (i, j, shift_tuple) -> EdgeGeometry.

    Parameters
    ----------
    atoms : ase.Atoms
        Current atomic configuration.
    cutoff : float
        Radial cutoff in Angstroms.

    Returns
    -------
    edges : dict
        {(i, j, (s0,s1,s2)): {'r_ij': np.ndarray, 'd_ij': float, 'rhat_ij': np.ndarray}}

    Notes
    -----
    The shift vector is in units of lattice vectors (integer).
    The physical displacement correction is shift @ cell.
    """
    from ase.neighborlist import primitive_neighbor_list

    pos = atoms.get_positions()
    cell = atoms.get_cell().array  # shape (3,3), rows are lattice vectors
    pbc = atoms.get_pbc()

    # primitive_neighbor_list returns arrays for all i, j, shift (image offsets)
    # self_interaction=False means i != j
    # bothways=True means both (i,j) and (j,i) are included (needed for GNN)
    i_indices, j_indices, offsets = primitive_neighbor_list(
        "ijS",
        pbc=pbc,
        cell=cell,
        positions=pos,
        cutoff=cutoff,
        self_interaction=False,
        use_scaled_positions=False,
    )

    edges = {}
    for idx in range(len(i_indices)):
        i = int(i_indices[idx])
        j = int(j_indices[idx])
        shift = tuple(int(x) for x in offsets[idx])

        r_ij = minimum_image_displacement(pos[i], pos[j], cell, np.array(offsets[idx]))
        d_ij, rhat_ij = compute_edge_geometry(r_ij)

        key = (i, j, shift)
        edges[key] = {
            "r_ij": r_ij,
            "d_ij": d_ij,
            "rhat_ij": rhat_ij,
        }

    return edges


def delta_r(
    r_current: np.ndarray,
    r_reference: np.ndarray,
) -> float:
    """
    Compute ||r_current - r_reference||.

    This is the primary geometric perturbation metric for E1.
    r_current and r_reference are full displacement vectors (3,), not positions.
    Both must already be MIC-corrected relative to their respective cells.

    NOTE: When the cell has deformed (e.g., NPT), the reference vector
    must be re-expressed in the current cell before computing this difference.
    For NVT (fixed cell) as used in E1, cells are identical, so this is safe.
    """
    diff = r_current - r_reference
    return float(np.linalg.norm(diff))


def delta_rhat(
    rhat_current: np.ndarray,
    rhat_reference: np.ndarray,
) -> float:
    """
    Compute ||rhat_current - rhat_reference||.

    This is the angular perturbation metric for E1.
    Both inputs must be unit vectors (shape (3,)).

    The maximum possible value is 2.0 (antiparallel unit vectors).
    For small rotations: ||Δrhat|| ≈ sin(angle) ≈ angle (radians).
    """
    diff = rhat_current - rhat_reference
    return float(np.linalg.norm(diff))


def is_dirty_zeroth_order(
    d_current: float,
    rhat_current: np.ndarray,
    d_reference: float,
    rhat_reference: np.ndarray,
    epsilon: float,
    sensitivity_factor: float = 1.0,
) -> bool:
    """
    Zeroth-order invalidation predicate (Frost document variant 0).

    An edge is DIRTY if its geometric perturbation × sensitivity exceeds epsilon.
    Uses the full displacement vector change as the primary signal.

    For E1 (measurement only), sensitivity_factor = 1.0 (no learned sensitivity).
    The full Frost system uses a per-species sensitivity lookup.

    Parameters
    ----------
    d_current, d_reference : float
        Current and reference scalar distances.
    rhat_current, rhat_reference : np.ndarray, shape (3,)
        Current and reference unit direction vectors.
    epsilon : float
        Force-error tolerance in meV/Angstrom.
    sensitivity_factor : float
        Edge sensitivity (default 1.0 for E1 measurement).

    Returns
    -------
    bool
        True if edge is DIRTY (should be recomputed).

    Notes
    -----
    Frost doc defines the predicate based on ||Delta r_ij|| where
    r_ij is the full MIC displacement vector. We reconstruct this
    from (d, rhat) as:
        r_current = d_current * rhat_current
        r_reference = d_reference * rhat_reference
        ||Delta r_ij|| = ||r_current - r_reference||
    """
    r_current = d_current * rhat_current
    r_reference = d_reference * rhat_reference
    dr = delta_r(r_current, r_reference)
    return (sensitivity_factor * dr) > epsilon


def compute_all_perturbations_batch(
    edges_current: dict,
    edges_reference: dict,
) -> dict:
    """
    Compute Δr and Δrhat for all edges that exist in both current and reference.

    Returns a dict: {edge_key: {'delta_r': float, 'delta_rhat': float, 'r_ij': np.ndarray}}
    for edges present in both snapshots.

    Edges that appear in current but not reference are NEW (must be dirty).
    Edges that appear in reference but not current have been removed.
    """
    results = {}
    for key, curr in edges_current.items():
        if key in edges_reference:
            ref = edges_reference[key]
            r_curr = curr["d_ij"] * curr["rhat_ij"]
            r_ref = ref["d_ij"] * ref["rhat_ij"]
            results[key] = {
                "delta_r": delta_r(r_curr, r_ref),
                "delta_rhat": delta_rhat(curr["rhat_ij"], ref["rhat_ij"]),
                "r_ij": curr["r_ij"],
                "d_ij": curr["d_ij"],
                "rhat_ij": curr["rhat_ij"],
                "new": False,
            }
        else:
            # Edge exists now but didn't in reference — treat as new/dirty
            results[key] = {
                "delta_r": float("inf"),
                "delta_rhat": float("inf"),
                "r_ij": curr["r_ij"],
                "d_ij": curr["d_ij"],
                "rhat_ij": curr["rhat_ij"],
                "new": True,
            }
    return results
