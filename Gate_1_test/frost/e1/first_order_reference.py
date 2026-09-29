"""Small real-MACE first-order Frost reference validation.

This is a bounded correctness diagnostic, not a production trajectory. It uses
an explicit dense Jacobian for a small edge subset and leaves all other MACE
edges exact. No cache lifecycle or production protocol is changed here.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from frost.e1.force_error import (
    compute_cached_forces,
    compute_exact_forces,
    extract_layer1_dense_jacobian,
    extract_layer1_local_inputs,
    extract_layer1_messages,
    layer1_edge_jvp,
    layer1_edge_vjp,
    make_layer1_edge_action,
)
from frost.e1.model_interface import load_model
from frost.e1.systems import build_system


def summarize_force_error(label, exact_energy, exact_forces, frost_energy, frost_forces):
    delta_force = frost_forces - exact_forces
    print(
        label,
        "exact_energy=%.15g" % exact_energy,
        "frost_energy=%.15g" % frost_energy,
        "abs_energy_error=%.6e" % abs(frost_energy - exact_energy),
        "max_force_error=%.6e" % np.abs(delta_force).max(),
        "mean_force_error=%.6e" % np.abs(delta_force).mean(),
        "rms_force_error=%.6e" % np.sqrt(np.mean(delta_force**2)),
        "relative_force_error=%.6e"
        % (np.linalg.norm(delta_force) / max(np.linalg.norm(exact_forces), 1e-30)),
        "exact_force_norm=%.6e" % np.linalg.norm(exact_forces),
        "frost_force_norm=%.6e" % np.linalg.norm(frost_forces),
        "finite=%s" % bool(np.isfinite(frost_energy) and np.isfinite(frost_forces).all()),
    )


def run(edge_limit: int, displacement: float) -> None:
    interface = load_model("mace-mp-0", device="cuda", dtype="float64")
    reference = build_system("diamond_si")

    local = extract_layer1_local_inputs(interface, reference)
    start = time.perf_counter()
    dense = extract_layer1_dense_jacobian(
        interface, reference, edge_limit=edge_limit, chunk_size=edge_limit
    )
    jacobian_seconds = time.perf_counter() - start
    g_ref = local["mji"].detach()
    vectors_ref = local["vectors"].detach()
    jacobian = torch.zeros(
        g_ref.shape[0], g_ref.shape[1], 3,
        device=g_ref.device,
        dtype=g_ref.dtype,
    )
    jacobian[:edge_limit] = dense["J_ref"]
    clean_mask = torch.zeros(g_ref.shape[0], dtype=torch.bool)
    clean_mask[:edge_limit] = True
    action_fns = [None] * g_ref.shape[0]
    for edge in range(edge_limit):
        action_fns[edge] = make_layer1_edge_action(interface, local, edge)

    print("EDGE_COUNT", g_ref.shape[0])
    print("CACHED_EDGE_COUNT", edge_limit)
    print("MESSAGE_SHAPE", tuple(g_ref.shape))
    print("JACOBIAN_SHAPE", tuple(dense["J_ref"].shape))
    print("JACOBIAN_SECONDS", jacobian_seconds)
    print("JACOBIAN_DTYPE_DEVICE", dense["J_ref"].dtype, dense["J_ref"].device)
    print("JACOBIAN_FINITE", bool(torch.isfinite(dense["J_ref"]).all()))

    for magnitude in (1e-4, 1e-3, 1e-2):
        current = reference.copy()
        current.positions[0, 0] += magnitude
        messages, vectors, _ = extract_layer1_messages(interface, current)
        delta = vectors[:edge_limit].detach() - vectors_ref[:edge_limit]
        predicted = g_ref[:edge_limit] + torch.einsum(
            "efa,ea->ef", dense["J_ref"], delta
        )
        error = predicted - messages[:edge_limit].detach()
        direct = torch.stack([
            layer1_edge_jvp(interface, local, edge, delta[edge])
            for edge in range(edge_limit)
        ])
        direct_error = direct - torch.einsum(
            "efa,ea->ef", dense["J_ref"], delta
        )
        print(
            "JVP",
            magnitude,
            "max=%.6e" % error.abs().max(),
            "mean=%.6e" % error.abs().mean(),
            "rms=%.6e" % torch.sqrt((error**2).mean()),
            "relative=%.6e" % (
                torch.linalg.vector_norm(error)
                / torch.linalg.vector_norm(messages[:edge_limit])
            ),
            "direct_vs_dense=%.6e" % direct_error.abs().max(),
        )

    model = interface.calculator.models[0]
    block = model.interactions[0]
    node_attrs = local["node_attrs"]
    atomic_numbers = model.atomic_numbers
    upstream = torch.linspace(
        -0.3,
        0.4,
        4 * g_ref.shape[1],
        device=g_ref.device,
        dtype=g_ref.dtype,
    ).reshape(4, g_ref.shape[1])
    expected_vjp = torch.einsum("efa,ef->ea", dense["J_ref"][:4], upstream)
    actual_vjp = []
    for edge in range(4):
        edge_index = local["edge_index"][:, edge : edge + 1]
        source = local["node_feats"][edge_index[0, 0] : edge_index[0, 0] + 1]
        vector = vectors_ref[edge].detach().requires_grad_(True)
        edge_vector = vector.reshape(1, 3)
        edge_attrs = model.spherical_harmonics(edge_vector)
        lengths = torch.linalg.vector_norm(edge_vector, dim=-1, keepdim=True)
        edge_feats, cutoff = model.radial_embedding(
            lengths, node_attrs, edge_index, atomic_numbers
        )
        weights = block.conv_tp_weights(edge_feats)
        if cutoff is not None:
            weights = weights * cutoff
        output = block.conv_tp(source, edge_attrs, weights)[0]
        actual_vjp.append(
            torch.autograd.grad((output * upstream[edge]).sum(), vector)[0]
        )
    actual_vjp = torch.stack(actual_vjp)
    vjp_error = actual_vjp - expected_vjp
    direct_vjp = torch.stack([
        layer1_edge_vjp(interface, local, edge, upstream[edge])
        for edge in range(4)
    ])
    direct_vjp_error = direct_vjp - expected_vjp
    print(
        "VJP",
        "shape=%s" % (tuple(actual_vjp.shape),),
        "max=%.6e" % vjp_error.abs().max(),
        "mean=%.6e" % vjp_error.abs().mean(),
        "relative=%.6e"
        % (torch.linalg.vector_norm(vjp_error) / torch.linalg.vector_norm(actual_vjp)),
        "direct_vs_dense=%.6e" % direct_vjp_error.abs().max(),
    )

    current = reference.copy()
    current.positions[0, 0] += displacement
    exact_energy, exact_forces = compute_exact_forces(current, interface)
    frost_energy, frost_forces = compute_cached_forces(
        current,
        interface,
        clean_mask,
        g_ref,
        mode=3,
        vectors=vectors_ref,
        vectors_ref=vectors_ref,
        action_fns=action_fns,
    )
    summarize_force_error(
        "FIRST_ORDER_PERTURBED",
        exact_energy,
        exact_forces,
        frost_energy,
        frost_forces,
    )

    exact_energy, exact_forces = compute_exact_forces(reference, interface)
    frost_energy, frost_forces = compute_cached_forces(
        reference,
        interface,
        clean_mask,
        g_ref,
        mode=3,
        vectors=vectors_ref,
        vectors_ref=vectors_ref,
        action_fns=action_fns,
    )
    summarize_force_error(
        "FIRST_ORDER_ZERO_DISPLACEMENT",
        exact_energy,
        exact_forces,
        frost_energy,
        frost_forces,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-limit", type=int, default=16)
    parser.add_argument("--displacement", type=float, default=1e-3)
    args = parser.parse_args()
    run(args.edge_limit, args.displacement)
