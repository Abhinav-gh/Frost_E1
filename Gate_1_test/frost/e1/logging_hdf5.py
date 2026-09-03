"""
Frost E1 — logging_hdf5.py

Chunked HDF5 logging for E1 trajectory data.

Storage strategy:
  - Per-step aggregate statistics are stored (NOT raw per-edge rows).
  - Raw per-edge data is stored only for 100 force-error sample frames.
  - One HDF5 file per trajectory (model x system x temperature x seed).

Rationale:
  Raw per-edge storage for Si (64 atoms, ~350 edges, 10,000 steps) would
  be ~28 million records per tolerance, ~7 GB for the full 8-tolerance sweep.
  Instead we store per-step mean/std/quantiles of Δr and Δrhat, plus per-step
  dirty fraction for each epsilon. This reduces to ~10,000 × O(100 floats)
  ≈ ~8 MB per trajectory, fully sufficient for E1 plots.

HDF5 Structure:
  /metadata          (attributes: model, system, temperature, seed, ...)
  /steps/timestep    (int64 array, shape [N_steps])
  /steps/n_edges     (int64 array)
  /steps/mean_delta_r (float64 array)
  /steps/std_delta_r  (float64 array)
  /steps/p50_delta_r  (float64 array)
  /steps/p90_delta_r  (float64 array)
  /steps/p99_delta_r  (float64 array)
  /steps/mean_delta_rhat (float64 array)
  /steps/std_delta_rhat  (float64 array)
  /steps/p50_delta_rhat  (float64 array)
  /steps/p90_delta_rhat  (float64 array)
  /tolerances/eps_{N}/dirty_count   (int64 array)
  /tolerances/eps_{N}/dirty_fraction (float64 array)
  /force_error_frames/  (group with one subgroup per sampled frame)
    frame_{t}/exact_forces     (float64, shape [N_atoms, 3])
    frame_{t}/stale_forces     (float64, shape [N_atoms, 3])
    frame_{t}/force_error_per_atom  (float64, shape [N_atoms])
    frame_{t}/max_error        (float64 scalar)
    frame_{t}/mean_error       (float64 scalar)
    frame_{t}/rms_error        (float64 scalar)
    frame_{t}/epsilon          (float64 scalar)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def _eps_key(eps: float) -> str:
    """Convert epsilon float to a valid HDF5 group name."""
    return f"eps_{eps:.4f}".replace(".", "p")


class E1Logger:
    """
    Streaming HDF5 logger for E1 experiment results.

    Usage
    -----
    with E1Logger(path, metadata) as log:
        for step, stats in trajectory:
            log.log_step(stats)
        log.log_force_error_frame(...)
    """

    def __init__(
        self,
        filepath: Path,
        metadata: dict,
        tolerances: List[float],
        n_steps_estimate: int = 10000,
        chunk_size: int = 500,
    ) -> None:
        self.filepath = Path(filepath)
        self.metadata = metadata
        self.tolerances = tolerances
        self.n_steps_estimate = n_steps_estimate
        self.chunk_size = chunk_size
        self._file: Optional[h5py.File] = None
        self._step_count = 0
        self._buffers: Dict[str, List] = {}
        self._init_buffers()

    def _init_buffers(self) -> None:
        """Initialize in-memory write buffers."""
        self._buffers = {
            "timestep": [],
            "n_edges": [],
            "n_edges_new": [],
            "n_edges_removed": [],
            "mean_delta_r": [],
            "std_delta_r": [],
            "p50_delta_r": [],
            "p90_delta_r": [],
            "p99_delta_r": [],
            "mean_delta_rhat": [],
            "std_delta_rhat": [],
            "p50_delta_rhat": [],
            "p90_delta_rhat": [],
        }
        for eps in self.tolerances:
            k = _eps_key(eps)
            self._buffers[f"dirty_count_{k}"] = []
            self._buffers[f"dirty_fraction_{k}"] = []

    def __enter__(self) -> "E1Logger":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def open(self) -> None:
        """Open HDF5 file and write metadata."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.filepath, "w")

        # Write metadata as attributes
        meta_grp = self._file.create_group("metadata")
        for k, v in self.metadata.items():
            try:
                meta_grp.attrs[k] = v
            except TypeError:
                meta_grp.attrs[k] = str(v)

        # Create step datasets with maxshape for chunked extension
        steps_grp = self._file.create_group("steps")
        scalar_keys = [
            "timestep", "n_edges", "n_edges_new", "n_edges_removed",
            "mean_delta_r", "std_delta_r", "p50_delta_r", "p90_delta_r", "p99_delta_r",
            "mean_delta_rhat", "std_delta_rhat", "p50_delta_rhat", "p90_delta_rhat",
        ]
        for key in scalar_keys:
            dtype = np.int64 if key in ("timestep", "n_edges", "n_edges_new", "n_edges_removed") else np.float64
            steps_grp.create_dataset(
                key,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=(self.chunk_size,),
            )

        # Tolerance datasets
        tol_grp = self._file.create_group("tolerances")
        for eps in self.tolerances:
            k = _eps_key(eps)
            eps_grp = tol_grp.create_group(k)
            eps_grp.attrs["epsilon_meV_per_A"] = eps
            for metric in ("dirty_count", "dirty_fraction"):
                dtype = np.int64 if metric == "dirty_count" else np.float64
                eps_grp.create_dataset(
                    metric,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=dtype,
                    chunks=(self.chunk_size,),
                )

        # Force error frames group
        self._file.create_group("force_error_frames")
        self._file.flush()
        logger.debug("Opened HDF5 log: %s", self.filepath)

    def log_step(self, stats: dict) -> None:
        """
        Buffer one production step's statistics.

        Parameters
        ----------
        stats : dict
            Output from PerEdgeReferenceCache.step().
        """
        self._buffers["timestep"].append(stats["timestep"])
        self._buffers["n_edges"].append(stats["n_edges_current"])
        self._buffers["n_edges_new"].append(stats["n_edges_new"])
        self._buffers["n_edges_removed"].append(stats["n_edges_removed"])
        self._buffers["mean_delta_r"].append(stats["mean_delta_r"])
        self._buffers["std_delta_r"].append(stats["std_delta_r"])
        self._buffers["p50_delta_r"].append(stats["p50_delta_r"])
        self._buffers["p90_delta_r"].append(stats["p90_delta_r"])
        self._buffers["p99_delta_r"].append(stats["p99_delta_r"])
        self._buffers["mean_delta_rhat"].append(stats["mean_delta_rhat"])
        self._buffers["std_delta_rhat"].append(stats["std_delta_rhat"])
        self._buffers["p50_delta_rhat"].append(stats["p50_delta_rhat"])
        self._buffers["p90_delta_rhat"].append(stats["p90_delta_rhat"])

        for eps in self.tolerances:
            k = _eps_key(eps)
            self._buffers[f"dirty_count_{k}"].append(stats["dirty_counts"][eps])
            self._buffers[f"dirty_fraction_{k}"].append(stats["dirty_fractions"][eps])

        self._step_count += 1

        # Flush buffer every chunk_size steps
        if self._step_count % self.chunk_size == 0:
            self._flush_buffers()

    def _flush_buffers(self) -> None:
        """Write buffered data to HDF5 and clear buffers."""
        if not self._buffers["timestep"]:
            return

        steps_grp = self._file["steps"]
        scalar_keys = [
            "timestep", "n_edges", "n_edges_new", "n_edges_removed",
            "mean_delta_r", "std_delta_r", "p50_delta_r", "p90_delta_r", "p99_delta_r",
            "mean_delta_rhat", "std_delta_rhat", "p50_delta_rhat", "p90_delta_rhat",
        ]
        for key in scalar_keys:
            ds = steps_grp[key]
            arr = np.array(self._buffers[key])
            old_size = ds.shape[0]
            ds.resize(old_size + len(arr), axis=0)
            ds[old_size:] = arr

        tol_grp = self._file["tolerances"]
        for eps in self.tolerances:
            k = _eps_key(eps)
            eps_grp = tol_grp[k]
            for metric in ("dirty_count", "dirty_fraction"):
                ds = eps_grp[metric]
                arr = np.array(self._buffers[f"{metric}_{k}"])
                old_size = ds.shape[0]
                ds.resize(old_size + len(arr), axis=0)
                ds[old_size:] = arr

        self._file.flush()
        self._init_buffers()  # clear buffers
        logger.debug("Flushed HDF5 buffer at step %d", self._step_count)

    def log_force_error_frame(
        self,
        frame_timestep: int,
        epsilon: float,
        exact_forces: np.ndarray,
        stale_forces: np.ndarray,
        positions: np.ndarray,
        cell: np.ndarray,
    ) -> None:
        """
        Store force-error frame data for post-hoc analysis.

        Parameters
        ----------
        frame_timestep : int
            Production step index for this frame.
        epsilon : float
            Tolerance at which stale forces were computed.
        exact_forces : np.ndarray, shape (N_atoms, 3)
            Forces computed with exact current geometry.
        stale_forces : np.ndarray, shape (N_atoms, 3)
            Forces computed with simulated stale-cache geometry.
        positions : np.ndarray, shape (N_atoms, 3)
            Atom positions at this frame.
        cell : np.ndarray, shape (3, 3)
            Simulation cell at this frame.
        """
        frame_grp = self._file["force_error_frames"].create_group(
            f"frame_{frame_timestep}_eps_{_eps_key(epsilon)}"
        )
        frame_grp.attrs["timestep"] = frame_timestep
        frame_grp.attrs["epsilon"] = epsilon

        error_per_atom = np.linalg.norm(exact_forces - stale_forces, axis=1)

        frame_grp.create_dataset("exact_forces", data=exact_forces.astype(np.float64))
        frame_grp.create_dataset("stale_forces", data=stale_forces.astype(np.float64))
        frame_grp.create_dataset("positions", data=positions.astype(np.float64))
        frame_grp.create_dataset("cell", data=cell.astype(np.float64))
        frame_grp.create_dataset("force_error_per_atom", data=error_per_atom.astype(np.float64))
        frame_grp.attrs["max_error"] = float(np.max(error_per_atom))
        frame_grp.attrs["mean_error"] = float(np.mean(error_per_atom))
        frame_grp.attrs["rms_error"] = float(np.sqrt(np.mean(error_per_atom ** 2)))
        frame_grp.attrs["p50_error"] = float(np.percentile(error_per_atom, 50))
        frame_grp.attrs["p90_error"] = float(np.percentile(error_per_atom, 90))
        frame_grp.attrs["p99_error"] = float(np.percentile(error_per_atom, 99))

        self._file.flush()

    def close(self) -> None:
        """Flush remaining buffer and close file."""
        if self._file is not None:
            self._flush_buffers()
            # Write lifetime data if provided separately
            self._file.flush()
            self._file.close()
            self._file = None
            logger.info("Closed HDF5 log: %s", self.filepath)

    def log_cache_lifetime(self, lifetime_summary: dict, epsilon: float) -> None:
        """Store cache lifetime statistics for a given tolerance."""
        if self._file is None:
            return
        k = _eps_key(epsilon)
        grp = self._file["tolerances"][k]
        for stat_name, val in lifetime_summary.items():
            grp.attrs[f"lifetime_{stat_name}"] = val
        self._file.flush()
