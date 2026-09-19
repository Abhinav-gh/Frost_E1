"""Tiny derivative benchmark for the Frost first-order message update.

This intentionally does not import MACE or modify the cache. It compares the
current elementwise Jacobian construction with torch.func transforms, and then
benchmarks the two contractions actually used by the first-order formula:
J @ delta_r for the message update and J.T @ upstream for conservative forces.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import torch
from torch.func import jacfwd, jacrev, jvp, vjp, vmap


@dataclass
class Measurement:
    name: str
    seconds: float
    shape: tuple
    max_difference: float
    memory_bytes: int | None


def message_function(vectors: torch.Tensor) -> torch.Tensor:
    """Small edge-local map R^3 -> R^F, batched over the leading dimension."""
    x, y, z = vectors.unbind(-1)
    return torch.stack(
        (
            x,
            y,
            z,
            x.square(),
            y.square(),
            z.square(),
            x * y,
            y * z,
            z * x,
            torch.sin(x + y + z),
            torch.cos(x - y),
            torch.exp(0.1 * z),
        ),
        dim=-1,
    )


def timed(name, function, reference, memory_bytes=None) -> Measurement:
    gc.collect()
    start = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - start
    difference = float((result - reference).abs().max())
    return Measurement(name, elapsed, tuple(result.shape), difference, memory_bytes)


def elementwise_jacobian(vectors: torch.Tensor) -> torch.Tensor:
    variables = vectors.detach().requires_grad_(True)
    values = message_function(variables)
    jacobian = torch.zeros(
        values.shape[0], values.shape[1], variables.shape[1], dtype=values.dtype
    )
    for edge in range(values.shape[0]):
        for feature in range(values.shape[1]):
            jacobian[edge, feature] = torch.autograd.grad(
                values[edge, feature], variables, retain_graph=True
            )[0][edge]
    return jacobian


def main() -> None:
    torch.set_default_dtype(torch.float64)
    edges = 8
    features = 12
    vectors = torch.linspace(-0.4, 0.7, edges * 3).reshape(edges, 3)
    delta = torch.linspace(0.001, 0.008, edges * 3).reshape(edges, 3)
    upstream = torch.linspace(-0.7, 0.9, edges * features).reshape(edges, features)

    reference_values = message_function(vectors)
    reference_jacobian = jacrev(message_function)(vectors[0])
    reference_jacobian = vmap(jacrev(message_function))(vectors)
    reference_jvp = torch.einsum("efa,ea->ef", reference_jacobian, delta)
    reference_vjp = torch.einsum("efa,ef->ea", reference_jacobian, upstream)

    measurements = []
    measurements.append(
        timed(
            "elementwise_autograd_full_jacobian",
            lambda: elementwise_jacobian(vectors),
            reference_jacobian,
            edges * features * 3 * vectors.element_size(),
        )
    )
    measurements.append(
        timed(
            "vmap_jacrev_full_jacobian",
            lambda: vmap(jacrev(message_function))(vectors),
            reference_jacobian,
            edges * features * 3 * vectors.element_size(),
        )
    )
    measurements.append(
        timed(
            "vmap_jacfwd_full_jacobian",
            lambda: vmap(jacfwd(message_function))(vectors),
            reference_jacobian,
            edges * features * 3 * vectors.element_size(),
        )
    )

    def batched_jvp():
        def one_edge(vector, direction):
            return jvp(message_function, (vector,), (direction,))[1]

        return vmap(one_edge)(vectors, delta)

    measurements.append(
        timed(
            "batched_jvp_message_update",
            batched_jvp,
            reference_jvp,
            edges * features * vectors.element_size(),
        )
    )

    def batched_vjp():
        def one_edge(vector, cotangent):
            _, pullback = vjp(message_function, vector)
            return pullback(cotangent)[0]

        return vmap(one_edge)(vectors, upstream)

    measurements.append(
        timed(
            "batched_vjp_force_contraction",
            batched_vjp,
            reference_vjp,
            edges * 3 * vectors.element_size(),
        )
    )

    print(f"SHAPES edges={edges} features={features} input=(E,3) output=(E,F)")
    print("REFERENCE_FULL_JACOBIAN", tuple(reference_jacobian.shape))
    print("REFERENCE_JVP", tuple(reference_jvp.shape))
    print("REFERENCE_VJP", tuple(reference_vjp.shape))
    print("NOTE memory_bytes is output storage only, not peak autograd memory")
    for measurement in measurements:
        print(
            "RESULT",
            measurement.name,
            "seconds=%.9f" % measurement.seconds,
            "shape=%s" % (measurement.shape,),
            "max_difference=%.3e" % measurement.max_difference,
            "output_memory_bytes=%s" % measurement.memory_bytes,
        )


if __name__ == "__main__":
    main()
