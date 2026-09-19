"""
Frost E1 — force_error.py

Force-error experiment for E1.

The key question: if we froze an edge's expensive geometric computation
at its reference-time value, what force error would that stale reuse induce?

IMPORTANT DISTINCTION (from Frost spec Section 10):
  - Model-vs-DFT error: how accurate is MACE-MP-0 vs ab initio?
  - Stale-cache-vs-exact-model error: how much does stale reuse hurt MACE?

E1 measures the SECOND quantity ONLY.

Stale Cache Implementation Strategy
------------------------------------
MACE-MP-0 is a message-passing GNN that takes:
  Input: atomic positions, species, cell, PBC
  Output: energy, forces

The Frost paper's central claim is that the expensive edge-local
tensor-product factor (layer 1 of MACE) is edge-local and can be cached.

The valid comparison path keeps coordinates unchanged and intercepts MACE's
first-layer edge messages before aggregation. The legacy atom-freezing helpers
remain below only for historical analysis and must not be used as Frost
correctness evidence.
"""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms

from frost.e1.mace_patch import apply_frost_patch


def _eager_spherical_harmonics(vector, spherical_harmonics):
    """Match e3nn's l<=3 basis without its scripted kernel boundary."""
    import torch

    x = torch.nn.functional.normalize(vector, dim=-1) if spherical_harmonics.normalize else vector
    x_coord, y_coord, z_coord = x.unbind(-1)
    sh_0_0 = torch.ones_like(x_coord)
    sh_1_0, sh_1_1, sh_1_2 = x_coord, y_coord, z_coord
    y2 = y_coord.pow(2)
    x2z2 = x_coord.pow(2) + z_coord.pow(2)
    sh_2 = (
        math.sqrt(3.0) * x_coord * z_coord,
        math.sqrt(3.0) * x_coord * y_coord,
        y2 - 0.5 * x2z2,
        math.sqrt(3.0) * y_coord * z_coord,
        math.sqrt(3.0) / 2.0 * (z_coord.pow(2) - x_coord.pow(2)),
    )
    sh_3 = (
        math.sqrt(5.0 / 6.0) * (sh_2[0] * z_coord + sh_2[4] * x_coord),
        math.sqrt(5.0) * sh_2[0] * y_coord,
        math.sqrt(3.0 / 8.0) * (4.0 * y2 - x2z2) * x_coord,
        0.5 * y_coord * (2.0 * y2 - 3.0 * x2z2),
        math.sqrt(3.0 / 8.0) * z_coord * (4.0 * y2 - x2z2),
        math.sqrt(5.0) * sh_2[4] * y_coord,
        math.sqrt(5.0 / 6.0) * (sh_2[4] * z_coord - sh_2[0] * x_coord),
    )
    all_values = torch.stack(
        (sh_0_0, sh_1_0, sh_1_1, sh_1_2, *sh_2, *sh_3), dim=-1
    )
    ls = spherical_harmonics._ls_list
    if ls != list(range(max(ls) + 1)):
        all_values = torch.cat(
            [all_values[..., l * l : (l + 1) * (l + 1)] for l in ls], dim=-1
        )
    scales = []
    for l in ls:
        if spherical_harmonics.normalization == "integral":
            scale = math.sqrt(2 * l + 1) / math.sqrt(4 * math.pi)
        elif spherical_harmonics.normalization == "component":
            scale = math.sqrt(2 * l + 1)
        else:
            scale = 1.0
        scales.extend([scale] * (2 * l + 1))
    return all_values * torch.tensor(scales, dtype=all_values.dtype, device=all_values.device)


def extract_layer1_reference_cache(model_interface, atoms: Atoms) -> dict:
    """Extract MACE's internal layer-1 messages and vector Jacobians.

    The returned tensors follow MACE's internal edge ordering and can therefore
    be passed directly to :func:`compute_cached_forces`.
    """
    import torch

    messages, vectors, edge_index = extract_layer1_messages(model_interface, atoms)
    jacobian = torch.zeros(messages.shape[0], messages.shape[1], 3, device=messages.device, dtype=messages.dtype)
    for edge in range(messages.shape[0]):
        for feature in range(messages.shape[1]):
            jacobian[edge, feature] = torch.autograd.grad(
                messages[edge, feature], vectors, retain_graph=True
            )[0][edge]
    return {
        "g_ref": messages.detach(),
        "J_ref": jacobian.detach(),
        "vectors_ref": vectors.detach(),
        "edge_index": edge_index.detach(),
    }


def extract_layer1_messages(model_interface, atoms: Atoms) -> tuple:
    """Capture first-layer messages without computing any Jacobians."""
    import torch

    calculator = model_interface.calculator
    model = calculator.models[0]
    batch = calculator._clone_batch(calculator._atoms_to_batch(atoms))
    dtype = next(model.parameters()).dtype
    for key in batch.keys:
        value = batch[key]
        if torch.is_tensor(value) and torch.is_floating_point(value):
            batch[key] = value.to(dtype=dtype)
    batch["positions"].requires_grad_(True)
    state = {
        "active": True,
        "mode": 0,
        "clean_mask": torch.empty(0, dtype=torch.bool),
        "g_ref": torch.empty(0),
    }
    from frost.e1.mace_patch import capture_layer1_vectors
    with capture_layer1_vectors(model, state), apply_frost_patch(model, state):
        model(batch.to_dict(), compute_force=False)
    messages = state["mji_exact"]
    vectors = state["vectors"]
    return messages, vectors, state["edge_index"]


def extract_layer1_local_inputs(model_interface, atoms: Atoms) -> dict:
    """Capture the real first-layer local MACE inputs without derivatives."""
    import torch

    calculator = model_interface.calculator
    model = calculator.models[0]
    batch = calculator._clone_batch(calculator._atoms_to_batch(atoms))
    dtype = next(model.parameters()).dtype
    for key in batch.keys:
        value = batch[key]
        if torch.is_tensor(value) and torch.is_floating_point(value):
            batch[key] = value.to(dtype=dtype)
    state = {
        "active": True,
        "mode": 0,
        "capture_local_inputs": True,
        "clean_mask": torch.empty(0, dtype=torch.bool),
        "g_ref": torch.empty(0),
    }
    from frost.e1.mace_patch import capture_layer1_vectors
    with capture_layer1_vectors(model, state), apply_frost_patch(model, state):
        model(batch.to_dict(), compute_force=False)
    state["local_inputs"]["vectors"] = state["vectors"].detach()
    state["local_inputs"]["unit_shifts"] = batch["unit_shifts"].detach()
    return state["local_inputs"]


def extract_layer1_dense_jacobian(
    model_interface,
    atoms: Atoms,
    edge_limit: Optional[int] = None,
    chunk_size: int = 16,
) -> dict:
    """Reference dense per-edge Jacobian using vectorized ``jacrev``."""
    import torch
    from torch.func import jacrev, vmap

    local = extract_layer1_local_inputs(model_interface, atoms)
    model = model_interface.calculator.models[0]
    block = model.interactions[0]
    edge_count = local["vectors"].shape[0]
    limit = edge_count if edge_limit is None else min(edge_limit, edge_count)
    vectors = local["vectors"][:limit].detach()
    edge_index = local["edge_index"][:, :limit]
    source_features = local["node_feats"][edge_index[0]].detach()
    node_attrs = local["node_attrs"]
    atomic_numbers = model.atomic_numbers

    def edge_message(vector, source_feature, one_edge_index):
        vector = vector.reshape(1, 3)
        edge_attrs = _eager_spherical_harmonics(vector, model.spherical_harmonics)
        lengths = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        edge_feats, cutoff = model.radial_embedding(
            lengths, node_attrs, one_edge_index.reshape(2, 1), atomic_numbers
        )
        weights = block.conv_tp_weights(edge_feats)
        if cutoff is not None:
            weights = weights * cutoff
        return block.conv_tp(source_feature.reshape(1, -1), edge_attrs, weights)[0]

    jacobian_chunks = []
    message_chunks = []
    jacobian_fn = vmap(jacrev(edge_message, argnums=0))
    message_fn = vmap(edge_message)
    for start in range(0, limit, chunk_size):
        stop = min(start + chunk_size, limit)
        chunk_vectors = vectors[start:stop]
        chunk_sources = source_features[start:stop]
        chunk_indices = edge_index[:, start:stop].T
        jacobian_chunks.append(jacobian_fn(chunk_vectors, chunk_sources, chunk_indices))
        message_chunks.append(message_fn(chunk_vectors, chunk_sources, chunk_indices))
    jacobian = torch.cat(jacobian_chunks, dim=0)
    messages = torch.cat(message_chunks, dim=0)
    return {
        "g_ref": messages.detach(),
        "J_ref": jacobian.detach(),
        "vectors_ref": vectors.detach(),
        "edge_index": edge_index.detach(),
        "edge_count_total": edge_count,
    }


def _layer1_edge_message_function(model, block, local_inputs, edge: int):
    """Build the real MACE first-layer message function for one edge."""
    import torch

    edge_index = local_inputs["edge_index"][:, edge : edge + 1]
    source = local_inputs["node_feats"][edge_index[0, 0] : edge_index[0, 0] + 1].detach()
    node_attrs = local_inputs["node_attrs"]
    atomic_numbers = model.atomic_numbers

    def message(vector):
        vector = vector.reshape(1, 3)
        edge_attrs = _eager_spherical_harmonics(vector, model.spherical_harmonics)
        lengths = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        edge_feats, cutoff = model.radial_embedding(
            lengths, node_attrs, edge_index, atomic_numbers
        )
        weights = block.conv_tp_weights(edge_feats)
        if cutoff is not None:
            weights = weights * cutoff
        return block.conv_tp(source, edge_attrs, weights)[0]

    return message


def layer1_edge_jvp(model_interface, local_inputs: dict, edge: int, direction):
    """Compute one real-MACE edge JVP without materializing its Jacobian."""
    import torch
    from torch.autograd.functional import jvp

    model = model_interface.calculator.models[0]
    message = _layer1_edge_message_function(model, model.interactions[0], local_inputs, edge)
    vector = local_inputs["vectors"][edge].detach()
    direction = torch.as_tensor(direction, device=vector.device, dtype=vector.dtype)
    return jvp(message, vector, direction, create_graph=False, strict=True)[1]


def layer1_edge_vjp(model_interface, local_inputs: dict, edge: int, upstream):
    """Compute one real-MACE edge VJP without materializing its Jacobian."""
    import torch

    model = model_interface.calculator.models[0]
    message = _layer1_edge_message_function(model, model.interactions[0], local_inputs, edge)
    vector = local_inputs["vectors"][edge].detach().requires_grad_(True)
    upstream = torch.as_tensor(upstream, device=vector.device, dtype=vector.dtype)
    output = message(vector)
    return torch.autograd.grad((output * upstream).sum(), vector)[0]

logger = logging.getLogger(__name__)


def compute_exact_forces(
    atoms: Atoms,
    model_interface,
) -> Tuple[float, np.ndarray]:
    """
    Compute exact energy and forces using current geometry.

    Parameters
    ----------
    atoms : ase.Atoms
        Current configuration.
    model_interface
        Loaded MACE interface.

    Returns
    -------
    energy : float (eV)
    forces : np.ndarray, shape (N, 3) (eV/Angstrom)
    """
    energy, forces = model_interface.evaluate(atoms)
    assert np.all(np.isfinite(forces)), "Non-finite exact forces detected"
    assert np.isfinite(energy), "Non-finite exact energy detected"
    return energy, forces


def identify_stale_atoms(
    atoms_current: Atoms,
    reference_positions: np.ndarray,
    perturbations: dict,
    epsilon: float,
) -> np.ndarray:
    """
    Identify atoms to "freeze" at reference positions for the stale evaluation.

    Strategy: An atom i is "stale" if ALL its current edges are beyond epsilon.
    This is conservative (overestimates staleness at atom level).

    See module docstring for the full explanation of this approximation.

    Parameters
    ----------
    atoms_current : ase.Atoms
        Current configuration.
    reference_positions : np.ndarray, shape (N, 3)
        Reference positions (at cache initialization or last refresh).
    perturbations : dict
        Output from compute_all_perturbations_batch: {(i,j,shift): {...}}
    epsilon : float
        Tolerance in meV/Angstrom.

    Returns
    -------
    stale_atom_mask : np.ndarray, shape (N,), dtype bool
        True for atoms to be held at reference positions.
    """
    n_atoms = len(atoms_current)

    # Count: for each atom, how many edges are dirty vs clean
    dirty_edges_per_atom = np.zeros(n_atoms, dtype=int)
    total_edges_per_atom = np.zeros(n_atoms, dtype=int)

    for (i, j, shift), pert in perturbations.items():
        total_edges_per_atom[i] += 1
        total_edges_per_atom[j] += 1
        dr = pert["delta_r"]
        if np.isinf(dr) or dr > epsilon:
            dirty_edges_per_atom[i] += 1
            dirty_edges_per_atom[j] += 1

    # Atom is "stale" (we reuse its old position) if ALL its edges are clean
    stale_mask = (
        (total_edges_per_atom > 0) &
        (dirty_edges_per_atom == 0)
    )
    return stale_mask


def compute_stale_forces(
    atoms_current: Atoms,
    reference_positions: np.ndarray,
    perturbations: dict,
    epsilon: float,
    model_interface,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute forces using simulated stale-cache configuration.

    Constructs a modified configuration where stale atoms are at reference
    positions and evaluates forces with the model.

    Parameters
    ----------
    atoms_current : ase.Atoms
        Current configuration with exact positions.
    reference_positions : np.ndarray, shape (N, 3)
        Reference positions (from the simulated cache).
    perturbations : dict
        Edge perturbation data from the cache.
    epsilon : float
        Tolerance for stale identification.
    model_interface
        Loaded MACE interface.

    Returns
    -------
    stale_forces : np.ndarray, shape (N, 3)
        Forces computed on the stale-patched configuration.
    stale_mask : np.ndarray, shape (N,), dtype bool
        Which atoms were set to reference positions.
    """
    stale_mask = identify_stale_atoms(
        atoms_current, reference_positions, perturbations, epsilon
    )

    # Create modified atoms: stale atoms get reference positions
    atoms_stale = atoms_current.copy()
    current_pos = atoms_current.get_positions()
    stale_pos = current_pos.copy()
    stale_pos[stale_mask] = reference_positions[stale_mask]
    atoms_stale.set_positions(stale_pos)

    _, stale_forces = model_interface.evaluate(atoms_stale)
    assert np.all(np.isfinite(stale_forces)), "Non-finite stale forces detected"

    return stale_forces, stale_mask


def compute_cached_forces(
    atoms: Atoms,
    model_interface,
    clean_mask,
    g_ref,
    mode: int = 1,
    J_ref=None,
    vectors=None,
    vectors_ref=None,
) -> Tuple[float, np.ndarray]:
    """Evaluate MACE on unchanged coordinates with edge-local interception."""
    import torch

    # ASE otherwise reuses the prior exact result when positions are unchanged,
    # which would bypass the patched MACE forward entirely.
    model_interface.calculator.results = {}
    state = {
        "active": True,
        "mode": mode,
        "clean_mask": torch.as_tensor(clean_mask, dtype=torch.bool),
        "g_ref": torch.as_tensor(g_ref),
    }
    if mode == 2:
        if J_ref is None or vectors is None or vectors_ref is None:
            raise ValueError("First-order cache requires J_ref, vectors, and vectors_ref")
        state.update({
            "J_ref": torch.as_tensor(J_ref),
            "vectors": torch.as_tensor(vectors),
            "vectors_ref": torch.as_tensor(vectors_ref),
        })
    from frost.e1.mace_patch import capture_layer1_vectors
    with capture_layer1_vectors(model_interface._calculator.models[0], state), apply_frost_patch(model_interface._calculator.models[0], state):
        return model_interface.evaluate(atoms)


def compute_force_error_statistics(
    exact_forces: np.ndarray,
    stale_forces: np.ndarray,
) -> dict:
    """
    Compute per-atom force error statistics.

    E1 measures: error_i = ||F_exact_i - F_stale_i||

    Parameters
    ----------
    exact_forces : np.ndarray, shape (N, 3)
    stale_forces : np.ndarray, shape (N, 3)

    Returns
    -------
    stats : dict with keys:
        'error_per_atom': np.ndarray, shape (N,)
        'max_error': float
        'mean_error': float
        'rms_error': float
        'p50_error': float
        'p90_error': float
        'p99_error': float
        'n_atoms': int
    """
    error_per_atom = np.linalg.norm(exact_forces - stale_forces, axis=1)
    return {
        "error_per_atom": error_per_atom,
        "max_error": float(np.max(error_per_atom)),
        "mean_error": float(np.mean(error_per_atom)),
        "rms_error": float(np.sqrt(np.mean(error_per_atom ** 2))),
        "p50_error": float(np.percentile(error_per_atom, 50)),
        "p90_error": float(np.percentile(error_per_atom, 90)),
        "p99_error": float(np.percentile(error_per_atom, 99)),
        "n_atoms": len(error_per_atom),
    }


def run_force_error_experiment(
    frame_atoms: Atoms,
    frame_perturbations: dict,
    reference_positions: np.ndarray,
    tolerances: List[float],
    model_interface,
) -> Dict[float, dict]:
    """
    Run the full force-error experiment for one sampled frame.

    For each tolerance epsilon:
      1. Identify stale atoms based on epsilon.
      2. Compute exact forces.
      3. Compute stale forces (stale atoms at reference positions).
      4. Compute force error statistics.

    Exact forces are computed once and reused across all tolerances.

    Parameters
    ----------
    frame_atoms : ase.Atoms
        The sampled production frame (exact current geometry).
    frame_perturbations : dict
        Edge perturbations relative to reference (from cache).
    reference_positions : np.ndarray, shape (N, 3)
        Reference atom positions at last cache refresh.
    tolerances : List[float]
        List of epsilon values.
    model_interface
        Loaded MACE interface.

    Returns
    -------
    results : dict[epsilon -> force_error_stats_dict]
    """
    # Exact evaluation (once)
    exact_energy, exact_forces = compute_exact_forces(frame_atoms, model_interface)
    logger.debug("Exact forces computed: max=%.4f eV/A", np.max(np.linalg.norm(exact_forces, axis=1)))

    results = {}
    for eps in tolerances:
        stale_forces, stale_mask = compute_stale_forces(
            frame_atoms, reference_positions, frame_perturbations, eps, model_interface
        )
        stats = compute_force_error_statistics(exact_forces, stale_forces)
        stats["epsilon"] = eps
        stats["exact_energy"] = exact_energy
        stats["n_stale_atoms"] = int(np.sum(stale_mask))
        stats["stale_fraction"] = float(np.mean(stale_mask))
        stats["exact_forces"] = exact_forces
        stats["stale_forces"] = stale_forces
        results[eps] = stats
        logger.debug(
            "  eps=%.3f: stale_atoms=%d/%d, max_force_error=%.4f eV/A",
            eps, stats["n_stale_atoms"], stats["n_atoms"], stats["max_error"]
        )

    return results
