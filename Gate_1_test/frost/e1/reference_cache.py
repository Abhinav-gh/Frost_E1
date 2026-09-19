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

    This state intentionally mirrors the stage-1 Frost design more closely than
    the original geometry-only cache: each edge has its own reference geometry,
    an associated layer-1 cache value, and a validity/generation counter.
    """
    edge_id: Tuple
    ref_r_ij: np.ndarray
    ref_d_ij: float
    ref_rhat_ij: np.ndarray
    ref_timestep: int
    age: int = 0
    generation: int = 0
    valid: bool = True
    state: str = "VALID"
    refresh_count: int = 0
    g_ref: Optional[np.ndarray] = None
    J_ref: Optional[np.ndarray] = None
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
        self.valid = True
        self.state = "VALID"
        self.generation += 1
        self.refresh_count += 1


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
        self._removed_keys = set()
        self._timestep: int = 0
        self._n_steps: int = 0

    def initialize(self, edges: dict, timestep: int = 0) -> None:
        """
        Initialize cache from t=0 edge geometry.

        All edges start with their t=0 geometry as the reference.
        age=0 at initialization.
        """
        self._entries.clear()
        self._removed_keys.clear()
        self._timestep = timestep
        for key, geom in edges.items():
            self._entries[key] = EdgeCacheEntry(
                edge_id=key,
                ref_r_ij=geom["r_ij"].copy(),
                ref_d_ij=geom["d_ij"],
                ref_rhat_ij=geom["rhat_ij"].copy(),
                ref_timestep=timestep,
                age=0,
                generation=0,
                valid=True,
                state="VALID",
            )
        self._n_steps = 0

    def step(self, edges_current: dict, timestep: int) -> dict:
        """
        Process one MD step.

        This is the real per-edge state transition for the E1 harness. It tracks
        the current dirty mask for each tolerance, the generation counters, and
        the reference refresh semantics without mutating atomic coordinates.
        """
        self._timestep = timestep
        self._n_steps += 1

        reference_entries = self._entries_as_ref_dict()
        perturbations = compute_all_perturbations_batch(edges_current, reference_entries)

        current_keys = set(edges_current.keys())
        old_keys = set(self._entries.keys())
        n_removed = len(old_keys - current_keys)
        n_new = len(current_keys - old_keys)
        removed_keys = old_keys - current_keys
        new_keys = current_keys - old_keys

        delta_r_vals = []
        delta_rhat_vals = []
        for key, pert in perturbations.items():
            if not pert["new"] and not np.isinf(pert["delta_r"]):
                delta_r_vals.append(pert["delta_r"])
                delta_rhat_vals.append(pert["delta_rhat"])

        delta_r_arr = np.array(delta_r_vals) if delta_r_vals else np.array([0.0])
        delta_rhat_arr = np.array(delta_rhat_vals) if delta_rhat_vals else np.array([0.0])

        dirty_counts = {}
        dirty_keys_by_eps = {eps: set() for eps in self.tolerances}
        for key in new_keys:
            if key not in self._entries:
                geom = edges_current[key]
                self._entries[key] = EdgeCacheEntry(
                    edge_id=key,
                    ref_r_ij=geom["r_ij"].copy(),
                    ref_d_ij=geom["d_ij"],
                    ref_rhat_ij=geom["rhat_ij"].copy(),
                    ref_timestep=timestep,
                    age=0,
                    generation=0,
                    valid=True,
                    state="VALID",
                )
                self._entries[key].state = "NEW"

        refresh_keys = set(new_keys)
        for eps in self.tolerances:
            n_dirty = 0
            for key, pert in perturbations.items():
                geom = edges_current[key]
                is_new = pert["new"]

                if is_new:
                    n_dirty += 1
                    dirty_keys_by_eps[eps].add(key)
                else:
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
                        dirty_keys_by_eps[eps].add(key)
                        entry.state = "DIRTY"
                        refresh_keys.add(key)

            dirty_counts[eps] = n_dirty

        for key in refresh_keys:
            geom = edges_current[key]
            entry = self._entries[key]
            if key in new_keys:
                entry.ref_r_ij = geom["r_ij"].copy()
                entry.ref_d_ij = geom["d_ij"]
                entry.ref_rhat_ij = geom["rhat_ij"].copy()
                entry.ref_timestep = timestep
                entry.age = 0
                entry.valid = True
                entry.state = "VALID"
            else:
                entry.refresh(geom["r_ij"], geom["d_ij"], geom["rhat_ij"], timestep)

        for key in current_keys:
            if key in self._entries:
                if self._entries[key].ref_timestep != timestep:
                    self._entries[key].age += 1

        for key in removed_keys:
            self._removed_keys.add(key)
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
            "new_keys": set(new_keys),
            "removed_keys": set(removed_keys),
            "perturbations": perturbations,
            "dirty_counts": dirty_counts,
            "dirty_fractions": dirty_fractions,
            "dirty_keys_by_eps": dirty_keys_by_eps,
            "cache_hits": {
                eps: n_edges_current - dirty_counts[eps]
                for eps in self.tolerances
            },
            "cache_misses": {
                eps: dirty_counts[eps]
                for eps in self.tolerances
            },
            "refresh_counts": {
                eps: len(dirty_keys_by_eps[eps])
                for eps in self.tolerances
            },
            "generation_by_key": {
                key: entry.generation for key, entry in self._entries.items()
            },
            "valid_edges": set(self._entries.keys()),
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
