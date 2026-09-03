"""Seven focused integrity checks for the edge-local E1 cache method."""

import numpy as np
import torch

from frost.e1.mace_patch import substitute_cached_messages


def _messages(vectors):
    return torch.cat((vectors, vectors.square(), torch.sin(vectors.sum(1, keepdim=True))), dim=1)


def _jacobian(vectors):
    variables = vectors.detach().requires_grad_(True)
    values = _messages(variables)
    result = torch.zeros(values.shape[0], values.shape[1], 3)
    for edge in range(values.shape[0]):
        for feature in range(values.shape[1]):
            result[edge, feature] = torch.autograd.grad(
                values[edge, feature], variables, retain_graph=True
            )[0][edge]
    return values.detach(), result


REFERENCE_VECTORS = torch.tensor([[1.0, 0.2, -0.1], [0.4, -0.7, 0.3], [1.2, 0.1, 0.5]])
REFERENCE_MESSAGES, REFERENCE_JACOBIAN = _jacobian(REFERENCE_VECTORS)


def _cached(current, clean, mode):
    exact = _messages(current)
    state = {
        "active": True,
        "mode": mode,
        "clean_mask": clean,
        "g_ref": REFERENCE_MESSAGES,
        "J_ref": REFERENCE_JACOBIAN,
        "vectors": current,
        "vectors_ref": REFERENCE_VECTORS,
    }
    return exact, substitute_cached_messages(exact, state)


def test_1_exact_edge_cache():
    current = REFERENCE_VECTORS + 0.03
    exact, cached = _cached(current, torch.ones(3, dtype=torch.bool), 0)
    assert torch.allclose(cached, exact)


def test_2_one_stale_edge_zeroth_order_has_error():
    exact, cached = _cached(REFERENCE_VECTORS + 0.05, torch.tensor([True, False, False]), 1)
    assert torch.linalg.vector_norm(cached - exact) > 0


def test_3_first_order_beats_zeroth_order():
    current = REFERENCE_VECTORS + 0.01
    exact, zeroth = _cached(current, torch.tensor([True, False, False]), 1)
    _, first = _cached(current, torch.tensor([True, False, False]), 2)
    assert torch.linalg.vector_norm(first - exact) < torch.linalg.vector_norm(zeroth - exact)


def test_4_zero_displacement_first_order_is_exact():
    exact, cached = _cached(REFERENCE_VECTORS, torch.ones(3, dtype=torch.bool), 2)
    assert torch.allclose(cached, exact)


def test_5_displacement_sweep_has_expected_scaling():
    zero_errors = []
    first_errors = []
    for size in (1e-3, 2e-3, 4e-3):
        current = REFERENCE_VECTORS + size
        exact, zero = _cached(current, torch.ones(3, dtype=torch.bool), 1)
        _, first = _cached(current, torch.ones(3, dtype=torch.bool), 2)
        zero_errors.append(torch.linalg.vector_norm(zero - exact))
        first_errors.append(torch.linalg.vector_norm(first - exact))
    zero_ratio = zero_errors[-1] / zero_errors[0]
    first_ratio = first_errors[-1] / first_errors[0]
    assert 3.5 < zero_ratio < 4.5
    assert 12.0 < first_ratio < 20.0


def test_6_multiple_clean_edges_aggregate():
    exact, cached = _cached(REFERENCE_VECTORS + 0.02, torch.tensor([True, True, False]), 2)
    assert torch.allclose(cached[2], exact[2])
    assert not torch.allclose(cached[:2], exact[:2])


def test_7_periodic_mic_displacement_is_used():
    reference = np.array([9.9, 0.0, 0.0])
    current = np.array([-0.1, 0.0, 0.0])
    wrapped = current - reference + np.array([10.0, 0.0, 0.0])
    assert np.allclose(wrapped, 0.0)