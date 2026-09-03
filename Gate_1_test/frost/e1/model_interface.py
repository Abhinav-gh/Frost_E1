"""
Frost E1 — model_interface.py

Thin wrapper around MACE-MP-0 (and eventually Allegro) for the E1 experiment.

Responsibilities:
  1. Load the model checkpoint cleanly.
  2. Provide a consistent ASE calculator interface.
  3. Report the exact cutoff radius (needed for neighbor list construction).
  4. Expose a way to evaluate energy + forces on an ASE Atoms object.
  5. Document what the model's internal edge ordering is and how we match it.

IMPORTANT — Edge Ordering
--------------------------
MACE-MP-0 builds its own neighbor list internally during forward pass.
We do NOT assume our independently constructed neighbor list matches
MACE's internal ordering. Instead:

  - We use our own neighbor list (built with ASE's primitive_neighbor_list)
    with the SAME cutoff as MACE uses internally.
  - The E1 perturbation measurement is done on OUR edge list.
  - The force-error experiment does NOT manipulate MACE internals:
    it provides full atomic positions to MACE and gets forces back.

For the stale-cache force-error experiment (Section 10 of spec), we 
simulate staleness by providing perturbed positions where some atoms
are held at their reference-time positions. This approach is described
in detail in docs/e1_stale_cache_definition.md.

ALLEGRO STATUS: BLOCKED
------------------------
Allegro has no universal pretrained checkpoint. E1 begins with MACE-MP-0.
If/when an Allegro checkpoint becomes available, add AllegroInterface
following the same pattern as MACEInterface.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MACEInterface:
    """
    MACE-MP-0 interface for E1 experiments.

    Uses mace-torch's MACECalculator as the backend.
    The calculator can be attached to ASE Atoms for MD.
    """

    # Known MACE-MP-0 cutoff radius (from mace-mp documentation and empirical check)
    # This is the SAME cutoff we use to build our neighbor lists.
    MACE_MP0_CUTOFF = 6.0  # Angstrom (medium model default)

    def __init__(
        self,
        model_type: str = "medium",
        device: str = "cuda",
        dtype: str = "float64",
        dispersion: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        model_type : str
            MACE-MP-0 model size: 'small', 'medium', 'large'.
        device : str
            'cuda' or 'cpu'.
        dtype : str
            'float64' or 'float32'. float64 recommended for measurements.
        dispersion : bool
            Whether to include D3 dispersion correction.
        """
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        self.dispersion = dispersion
        self._calculator = None
        self._is_loaded = False

    def load(self) -> None:
        """
        Load MACE-MP-0. Downloads from HuggingFace on first call.

        Raises
        ------
        ImportError
            If mace-torch is not installed.
        RuntimeError
            If download fails or model is corrupted.
        """
        try:
            from mace.calculators import mace_mp
        except ImportError as e:
            raise ImportError(
                "mace-torch is not installed. "
                "Install with: pip install mace-torch\n"
                f"Original error: {e}"
            ) from e

        logger.info(
            "Loading MACE-MP-0 (%s, device=%s, dtype=%s)...",
            self.model_type,
            self.device,
            self.dtype,
        )

        default_dtype = torch.float64 if self.dtype == "float64" else torch.float32

        self._calculator = mace_mp(
            model=self.model_type,
            dispersion=self.dispersion,
            default_dtype=self.dtype,
            device=self.device,
        )
        self._is_loaded = True

        # Report actual cutoff
        try:
            cutoff = self._get_actual_cutoff()
            logger.info("MACE-MP-0 cutoff: %.3f Angstrom", cutoff)
            if abs(cutoff - self.MACE_MP0_CUTOFF) > 0.1:
                logger.warning(
                    "Actual cutoff %.3f differs from assumed %.3f; "
                    "update MACE_MP0_CUTOFF.",
                    cutoff, self.MACE_MP0_CUTOFF,
                )
        except Exception:
            logger.warning("Could not determine actual MACE cutoff; using %.1f", self.MACE_MP0_CUTOFF)

    def _get_actual_cutoff(self) -> float:
        """
        Try to extract the actual cutoff radius from the MACE model internals.
        Falls back to MACE_MP0_CUTOFF if introspection fails.
        """
        try:
            # mace_mp calculator wraps a MACE model
            models = self._calculator.models
            if models:
                m = models[0]
                # MACE stores r_max in model
                if hasattr(m, "r_max"):
                    return float(m.r_max)
        except Exception:
            pass
        return self.MACE_MP0_CUTOFF

    @property
    def cutoff(self) -> float:
        """Cutoff radius in Angstrom used for neighbor list construction."""
        if self._is_loaded:
            try:
                return self._get_actual_cutoff()
            except Exception:
                pass
        return self.MACE_MP0_CUTOFF

    @property
    def calculator(self):
        """Return the ASE calculator (attach to Atoms for MD)."""
        if not self._is_loaded:
            raise RuntimeError("Call load() first.")
        return self._calculator

    def evaluate(self, atoms) -> Tuple[float, np.ndarray]:
        """
        Evaluate energy and forces on atoms.

        Parameters
        ----------
        atoms : ase.Atoms
            Configuration to evaluate.

        Returns
        -------
        energy : float
            Potential energy in eV.
        forces : np.ndarray, shape (N, 3)
            Forces in eV/Angstrom.
        """
        if not self._is_loaded:
            raise RuntimeError("Call load() first.")
        atoms.calc = self._calculator
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces().copy()
        return energy, forces

    def __repr__(self) -> str:
        return (
            f"MACEInterface(model_type={self.model_type!r}, "
            f"device={self.device!r}, dtype={self.dtype!r}, "
            f"loaded={self._is_loaded})"
        )


class AllegroInterface:
    """
    PLACEHOLDER for Allegro interface.

    STATUS: BLOCKED — No public universal pretrained Allegro checkpoint exists.

    This class will be implemented when:
    1. A trained Allegro checkpoint for the E1 benchmark systems is available, OR
    2. A universal Allegro model analogous to MACE-MP-0 is released.

    Do NOT use this class in E1 runs until it has a real checkpoint.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, **kwargs) -> None:
        self.checkpoint_path = checkpoint_path

    def load(self) -> None:
        raise NotImplementedError(
            "ALLEGRO BLOCKER: No universal pretrained Allegro checkpoint available. "
            "E1 currently runs with MACE-MP-0 only. "
            "To enable Allegro: provide a trained checkpoint path and implement "
            "this interface using nequip's Trainer.load_model() API."
        )

    def evaluate(self, atoms) -> Tuple[float, np.ndarray]:
        raise NotImplementedError("Allegro not available. See AllegroInterface.load().")


def load_model(model_name: str, device: str = "cuda", dtype: str = "float64"):
    """
    Factory function. Returns a loaded model interface.

    Parameters
    ----------
    model_name : str
        'mace-mp-0' or 'allegro'.
    device : str
        'cuda' or 'cpu'.

    Returns
    -------
    Loaded model interface with .evaluate(), .cutoff, .calculator properties.
    """
    if model_name in ("mace-mp-0", "mace_mp_0", "mace"):
        interface = MACEInterface(model_type="medium", device=device, dtype=dtype)
        interface.load()
        return interface
    elif model_name in ("allegro",):
        raise NotImplementedError(
            "Allegro: BLOCKED. No universal pretrained checkpoint. "
            "See AllegroInterface docstring."
        )
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Supported: 'mace-mp-0'. ('allegro' is blocked — see docs)."
        )
