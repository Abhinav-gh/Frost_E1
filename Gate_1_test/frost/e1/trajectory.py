"""
Frost E1 — trajectory.py

NVT MD trajectory driver for E1 Redundancy Characterisation.

Protocol (from Frost document):
  - Thermostat: Langevin NVT
  - Friction: gamma = 0.01 fs^-1
  - Timestep: 1 fs
  - Equilibration: 20 ps (20,000 steps) — DISCARD
  - Production: 10,000 steps — LOG EVERY STEP
  - Fixed random seed

Seed strategy:
  - Velocity initialization: np.random.seed(seed) -> MaxwellBoltzmann
  - Langevin thermostat: seed used to initialize ASE's Langevin RNG
  - Recorded in output metadata for exact reproducibility

NOTE: We claim to follow the stated protocol, NOT to reproduce exact 
trajectories from the original paper. Trajectory outcome depends on
the model checkpoint, equilibration starting point, and integrator
implementation details that may differ.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
from ase import Atoms
from ase.md.langevin import Langevin
from ase.io import Trajectory as ASETrajectory
from ase import units

logger = logging.getLogger(__name__)


def initialize_velocities(atoms: Atoms, temperature_K: float, seed: int) -> None:
    """
    Initialize Maxwell-Boltzmann velocity distribution.

    Uses np.random.seed(seed) before sampling. The seed is recorded
    in the experiment metadata for reproducibility.

    Parameters
    ----------
    atoms : ase.Atoms
        Configuration to initialize velocities for.
    temperature_K : float
        Target temperature in Kelvin.
    seed : int
        Random seed for velocity initialization.
    """
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

    rng = np.random.default_rng(seed)
    # ASE's MaxwellBoltzmannDistribution uses numpy's global RNG internally
    # We set it explicitly
    np.random.seed(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    # Remove net linear and angular momentum
    Stationary(atoms)
    ZeroRotation(atoms)


def run_md_trajectory(
    atoms: Atoms,
    calculator,
    temperature_K: float,
    n_equilibration_steps: int,
    n_production_steps: int,
    timestep_fs: float,
    langevin_gamma: float,
    seed: int,
    step_callback: Optional[Callable[[Atoms, int], None]] = None,
    log_interval: int = 1,
) -> dict:
    """
    Run NVT Langevin MD and call step_callback on every production step.

    Parameters
    ----------
    atoms : ase.Atoms
        Starting configuration. Modified in-place.
    calculator
        Loaded model interface (or ASE calculator).
    temperature_K : float
        Target temperature in K.
    n_equilibration_steps : int
        Number of NVT steps to discard (equilibration).
    n_production_steps : int
        Number of NVT steps to log (production).
    timestep_fs : float
        MD timestep in femtoseconds.
    langevin_gamma : float
        Langevin friction coefficient in 1/fs (from Frost doc: 0.01 fs^-1).
    seed : int
        Random seed for velocity init and thermostat.
    step_callback : callable or None
        Called with (atoms, production_step_index) at every production step.
        atoms contains the current positions/velocities/forces.
    log_interval : int
        How often (production steps) to invoke callback. Default 1 (every step).

    Returns
    -------
    metadata : dict
        Run information (timings, step counts, seed, final energy, etc.)
    """
    # Attach calculator
    if hasattr(calculator, "calculator"):
        atoms.calc = calculator.calculator
    else:
        atoms.calc = calculator

    # Initialize velocities
    initialize_velocities(atoms, temperature_K, seed)

    # Convert timestep: ASE uses internal units (fs is fine with units.fs)
    dt_ase = timestep_fs * units.fs  # ASE internal time unit

    # Langevin gamma: ASE expects units of 1/time_units
    # friction = gamma [1/fs] * units.fs -> dimensionless in ASE units
    friction_ase = langevin_gamma / units.fs

    # Build Langevin integrator
    # rng for Langevin is seeded separately
    dyn = Langevin(
        atoms,
        timestep=dt_ase,
        temperature_K=temperature_K,
        friction=friction_ase,
        # ASE Langevin uses numpy RNG internally
        # We seed numpy globally before constructing the integrator
    )
    np.random.seed(seed + 1000)  # offset from velocity seed to avoid correlation

    # ---- EQUILIBRATION PHASE ----
    logger.info(
        "Starting equilibration: %d steps at T=%.0f K (seed=%d)",
        n_equilibration_steps,
        temperature_K,
        seed,
    )
    t_eq_start = time.perf_counter()

    if n_equilibration_steps > 0:
        dyn.run(n_equilibration_steps)

    t_eq_end = time.perf_counter()
    eq_wall_time = t_eq_end - t_eq_start
    logger.info(
        "Equilibration complete: %.2f s (%.2f ms/step)",
        eq_wall_time,
        1000 * eq_wall_time / max(n_equilibration_steps, 1),
    )

    # ---- PRODUCTION PHASE ----
    logger.info(
        "Starting production: %d steps at T=%.0f K",
        n_production_steps,
        temperature_K,
    )
    t_prod_start = time.perf_counter()

    production_energies = []
    production_temps = []

    def _production_callback() -> None:
        """Internal callback: log energy/temp, invoke user callback."""
        step_idx = dyn.nsteps - n_equilibration_steps - 1
        if step_idx < 0:
            return

        energy = float(atoms.get_potential_energy())
        temperature = float(atoms.get_temperature())
        production_energies.append(energy)
        production_temps.append(temperature)

        if step_callback is not None and step_idx % log_interval == 0:
            step_callback(atoms, step_idx)

        if step_idx % 1000 == 0 and step_idx > 0:
            logger.info(
                "  Production step %5d / %d | E=%.4f eV | T=%.1f K",
                step_idx,
                n_production_steps,
                energy,
                temperature,
            )

    dyn.attach(_production_callback, interval=1)
    dyn.run(n_production_steps)

    t_prod_end = time.perf_counter()
    prod_wall_time = t_prod_end - t_prod_start

    # Compute energy drift (NVE would be tighter; NVT is just for reference)
    energies_arr = np.array(production_energies)
    temps_arr = np.array(production_temps)

    metadata = {
        "n_atoms": len(atoms),
        "temperature_K": temperature_K,
        "n_equilibration_steps": n_equilibration_steps,
        "n_production_steps": n_production_steps,
        "timestep_fs": timestep_fs,
        "langevin_gamma": langevin_gamma,
        "seed": seed,
        "eq_wall_time_s": eq_wall_time,
        "prod_wall_time_s": prod_wall_time,
        "ms_per_step": 1000 * prod_wall_time / max(n_production_steps, 1),
        "mean_energy_eV": float(np.mean(energies_arr)) if len(energies_arr) else None,
        "std_energy_eV": float(np.std(energies_arr)) if len(energies_arr) else None,
        "mean_temperature_K": float(np.mean(temps_arr)) if len(temps_arr) else None,
        "final_energy_eV": float(energies_arr[-1]) if len(energies_arr) else None,
    }

    logger.info(
        "Production complete: %.2f s (%.2f ms/step) | "
        "mean E=%.4f eV | mean T=%.1f K",
        prod_wall_time,
        metadata["ms_per_step"],
        metadata["mean_energy_eV"] or 0,
        metadata["mean_temperature_K"] or 0,
    )

    return metadata
