"""
Frost E1 — reference_cache.py

Simulated per-edge reference cache for E1 Redundancy Characterisation.

This is NOT a production Frost cache. It is an instrumented simulation
of what a per-edge geometric cache would look like, for the purpose of
measuring temporal redundancy. No speedup is implemented here.

Key design principles:
  - Each edge has its own reference geometry and reference timestep.
  - References are updated independently per edge (not globally every k steps).
  - The "dirty" predicate is evaluated per-epsilon, per-edge.
  - Cache lifetime (number of steps between reference refreshes) is tracked.

State machine per edge:
    NEW -> VALID (first compute) -> DIRTY (predicate triggered) -> VALID
    VALID -> REMOVED (edge leaves neighbor list)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from frost.e1.geometry import (
    compute_all_perturbations_batch,
    is_dirty_zeroth_order,
)


@dataclass
class EdgeCacheEntry:
    """
    Per-edge cache record for the simulated E1 cache.

    Attributes
    ----------
    ref_r_ij : np.ndarray, shape (3,)
        Reference displacement vector (MIC-corrected) at last refresh.
    ref_d_ij : float
        Reference scalar distance at last refresh.
    ref_rhat_ij : np.ndarray, shape (3,)
        Reference unit direction at last refresh.
    ref_timestep : int
        Simulation timestep at which reference was set.
    age : int
        Number of steps since last reference refresh.
    lifetime_history : List[int]
        History of consecutive valid lifetimes (steps between refreshes).
        Populated each time the entry is refreshed after being dirty.
    """
    ref_r_ij: np.ndarray
    ref_d_ij: float
    ref_rhat_ij: np.ndarray
    ref_timestep: int
    age: int = 0
    lifetime_history: List[int] = field(default_factory=list)

    def refresh(self, r_ij: np.ndarray, d_ij: float, rhat_ij: np.ndarray, timestep: int) -> None:
        """Update reference geometry. Records the completed lifetime."""
        if self.age > 0:
            self.lifetime_history.append(self.age)
        self.ref_r_ij = r_ij.copy()
        self.ref_d_ij = d_ij
        self.ref_rhat_ij = rhat_ij.copy()
        self.ref_timestep = timestep
        self.age = 0


class PerEdgeReferenceCache:
    """
    Simulated per-edge reference cache for E1 measurements.

    Usage
    -----
    cache = PerEdgeReferenceCache(tolerances=[0.1, 1.0, 10.0])
    cache.initialize(edges_t0, timestep=0)

    for t, edges_t in enumerate(trajectory):
        step_stats = cache.step(edges_t, timestep=t+1)
        # step_stats contains dirty fractions for each epsilon, etc.
    """

    def __init__(self, tolerances: List[float]) -> None:
        """
        Parameters
        ----------
        tolerances : List[float]
            List of epsilon values (meV/Angstrom) to sweep.
        """
        self.tolerances = sorted(tolerances)
        self._entries: Dict[Tuple, EdgeCacheEntry] = {}
        self._timestep: int = 0
        self._n_steps: int = 0

    def initialize(self, edges: dict, timestep: int = 0) -> None:
        """
        Initialize cache from t=0 edge geometry.

        All edges start with their t=0 geometry as the reference.
        age=0 at initialization.
        """
        self._entries.clear()
        self._timestep = timestep
        for key, geom in edges.items():
            self._entries[key] = EdgeCacheEntry(
                ref_r_ij=geom["r_ij"].copy(),
                ref_d_ij=geom["d_ij"],
                ref_rhat_ij=geom["rhat_ij"].copy(),
                ref_timestep=timestep,
                age=0,
            )
        self._n_steps = 0

    def step(self, edges_current: dict, timestep: int) -> dict:
        """
        Process one MD step.

        For every current edge:
          1. Compute Δr and Δrhat relative to its reference.
          2. For each tolerance epsilon, classify as dirty or clean.
          3. If dirty at epsilon: update reference for that edge.

        NOTE: The reference update is per-tolerance. This means each edge
        has multiple reference states, one per epsilon. This is correct for
        measuring the dirty fraction independently at each tolerance.

        Parameters
        ----------
        edges_current : dict
            Current edge geometry from build_edge_geometry_from_ase().
        timestep : int
            Current simulation timestep.

        Returns
        -------
        stats : dict with keys:
            'timestep': int
            'n_edges_current': int
            'n_edges_new': int       # edges not in previous reference
            'n_edges_removed': int   # edges that disappeared
            'perturbations': dict of {key: {'delta_r': float, 'delta_rhat': float}}
            'dirty_counts': dict of {epsilon: int}  # n dirty edges per epsilon
            'dirty_fractions': dict of {epsilon: float}
            'mean_delta_r': float
            'std_delta_r': float
            'p50_delta_r': float
            'p90_delta_r': float
            'p99_delta_r': float
            'mean_delta_rhat': float
        """
        self._timestep = timestep
        self._n_steps += 1

        # Compute perturbations for all current edges
        perturbations = compute_all_perturbations_batch(edges_current, self._entries_as_ref_dict())

        # Count edges that no longer exist (removed from neighbor list)
        current_keys = set(edges_current.keys())
        old_keys = set(self._entries.keys())
        n_removed = len(old_keys - current_keys)
        n_new = len(current_keys - old_keys)

        # Collect Δr / Δrhat arrays for statistics
        delta_r_vals = []
        delta_rhat_vals = []
        for key, pert in perturbations.items():
            if not pert["new"] and not np.isinf(pert["delta_r"]):
                delta_r_vals.append(pert["delta_r"])
                delta_rhat_vals.append(pert["delta_rhat"])

        delta_r_arr = np.array(delta_r_vals) if delta_r_vals else np.array([0.0])
        delta_rhat_arr = np.array(delta_rhat_vals) if delta_rhat_vals else np.array([0.0])

        # Per-tolerance: compute dirty count and update references
        dirty_counts = {}
        for eps in self.tolerances:
            n_dirty = 0
            for key, pert in perturbations.items():
                geom = edges_current[key]
                is_new = pert["new"]

                if is_new:
                    # New edge: always dirty (must compute fresh)
                    n_dirty += 1
                    # Create new cache entry for this edge
                    if key not in self._entries:
                        self._entries[key] = EdgeCacheEntry(
                            ref_r_ij=geom["r_ij"].copy(),
                            ref_d_ij=geom["d_ij"],
                            ref_rhat_ij=geom["rhat_ij"].copy(),
                            ref_timestep=timestep,
                            age=0,
                        )
                else:
                    # Existing edge: check predicate
                    entry = self._entries[key]
                    dirty = is_dirty_zeroth_order(
                        d_current=geom["d_ij"],
                        rhat_current=geom["rhat_ij"],
                        d_reference=entry.ref_d_ij,
                        rhat_reference=entry.ref_rhat_ij,
                        epsilon=eps,
                    )
                    if dirty:
                        n_dirty += 1
                        # Refresh reference for this edge
                        entry.refresh(geom["r_ij"], geom["d_ij"], geom["rhat_ij"], timestep)

            dirty_counts[eps] = n_dirty

        # Age increment for all surviving edges
        for key in current_keys:
            if key in self._entries:
                if self._entries[key].ref_timestep != timestep:
                    self._entries[key].age += 1

        # Remove stale entries for disappeared edges
        for key in old_keys - current_keys:
            del self._entries[key]

        n_edges_current = len(current_keys)
        dirty_fractions = {
            eps: dirty_counts[eps] / max(n_edges_current, 1)
            for eps in self.tolerances
        }

        return {
            "timestep": timestep,
            "n_edges_current": n_edges_current,
            "n_edges_new": n_new,
            "n_edges_removed": n_removed,
            "perturbations": perturbations,
            "dirty_counts": dirty_counts,
            "dirty_fractions": dirty_fractions,
            "mean_delta_r": float(np.mean(delta_r_arr)),
            "std_delta_r": float(np.std(delta_r_arr)),
            "p50_delta_r": float(np.percentile(delta_r_arr, 50)),
            "p90_delta_r": float(np.percentile(delta_r_arr, 90)),
            "p99_delta_r": float(np.percentile(delta_r_arr, 99)),
            "mean_delta_rhat": float(np.mean(delta_rhat_arr)),
            "std_delta_rhat": float(np.std(delta_rhat_arr)),
            "p50_delta_rhat": float(np.percentile(delta_rhat_arr, 50)),
            "p90_delta_rhat": float(np.percentile(delta_rhat_arr, 90)),
        }

    def get_lifetime_summary(self) -> dict:
        """
        Return cache lifetime statistics across all entries.

        'cache lifetime' = number of consecutive steps an edge was valid
        before being declared dirty and refreshed.
        """
        all_lifetimes = []
        for entry in self._entries.values():
            all_lifetimes.extend(entry.lifetime_history)
            if entry.age > 0:
                all_lifetimes.append(entry.age)  # current in-progress run

        if not all_lifetimes:
            return {}

        arr = np.array(all_lifetimes, dtype=float)
        return {
            "n_refresh_events": len(arr),
            "mean_lifetime": float(np.mean(arr)),
            "median_lifetime": float(np.median(arr)),
            "p50_lifetime": float(np.percentile(arr, 50)),
            "p90_lifetime": float(np.percentile(arr, 90)),
            "p99_lifetime": float(np.percentile(arr, 99)),
            "max_lifetime": float(np.max(arr)),
            "min_lifetime": float(np.min(arr)),
        }

    def _entries_as_ref_dict(self) -> dict:
        """Convert cache entries to the format expected by compute_all_perturbations_batch."""
        return {
            key: {
                "r_ij": entry.ref_r_ij,
                "d_ij": entry.ref_d_ij,
                "rhat_ij": entry.ref_rhat_ij,
            }
            for key, entry in self._entries.items()
        }

    @property
    def n_edges(self) -> int:
        return len(self._entries)
