import torch

from frost.e1.force_error import make_first_order_edge_action
from frost.e1.mace_patch import substitute_cached_messages


def message_map(vector):
    return torch.stack(
        (vector[0] + vector[1] ** 2, vector[1] * vector[2], torch.sin(vector[0] - vector[2]))
    )


def dense_jacobian(reference):
    return torch.autograd.functional.jacobian(message_map, reference)


def test_jvp_matches_dense_jacobian_action():
    reference = torch.tensor([0.7, -0.2, 0.4], dtype=torch.float64)
    direction = torch.tensor([0.03, -0.04, 0.02], dtype=torch.float64)
    jacobian = dense_jacobian(reference)
    expected = jacobian @ direction
    actual = torch.autograd.functional.jvp(message_map, reference, direction)[1]
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_vjp_matches_dense_jacobian_transpose_action():
    reference = torch.tensor([0.7, -0.2, 0.4], dtype=torch.float64)
    upstream = torch.tensor([0.2, -0.5, 0.8], dtype=torch.float64)
    jacobian = dense_jacobian(reference)
    expected = jacobian.T @ upstream
    variable = reference.detach().requires_grad_(True)
    actual = torch.autograd.grad((message_map(variable) * upstream).sum(), variable)[0]
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_action_zero_displacement_matches_reference_message():
    reference = torch.tensor([0.7, -0.2, 0.4], dtype=torch.float64)
    reference_message = message_map(reference).detach()
    jacobian = dense_jacobian(reference)
    action = make_first_order_edge_action(
        lambda direction: jacobian @ direction,
        lambda upstream: jacobian.T @ upstream,
    )
    current = reference.detach().requires_grad_(True)
    actual = action(current, reference_message, reference)
    assert torch.allclose(actual, reference_message, atol=1e-12, rtol=1e-12)


def test_frost_energy_gradient_matches_finite_difference_with_fixed_reference():
    reference = torch.tensor([0.7, -0.2, 0.4], dtype=torch.float64)
    current = torch.tensor([0.73, -0.24, 0.42], dtype=torch.float64, requires_grad=True)
    reference_message = message_map(reference).detach()
    jacobian = dense_jacobian(reference)
    upstream = torch.tensor([0.2, -0.5, 0.8], dtype=torch.float64)
    action = make_first_order_edge_action(
        lambda direction: jacobian @ direction,
        lambda incoming: jacobian.T @ incoming,
    )

    surrogate = (action(current, reference_message, reference) * upstream).sum()
    gradient = torch.autograd.grad(surrogate, current)[0]

    finite_difference = []
    step = 1e-6
    for coordinate in range(3):
        plus = current.detach().clone()
        minus = current.detach().clone()
        plus[coordinate] += step
        minus[coordinate] -= step
        plus_value = (action(plus, reference_message, reference) * upstream).sum()
        minus_value = (action(minus, reference_message, reference) * upstream).sum()
        finite_difference.append((plus_value - minus_value) / (2 * step))
    finite_difference = torch.stack(finite_difference)
    assert torch.allclose(gradient, finite_difference, atol=1e-8, rtol=1e-6)


def test_mode_three_substitution_uses_only_clean_edge_actions():
    reference = torch.tensor([0.7, -0.2, 0.4], dtype=torch.float64)
    current = torch.stack((reference + 0.03, reference + 0.05))
    exact = torch.stack((message_map(current[0]), message_map(current[1])))
    references = torch.stack((message_map(reference), message_map(reference))).detach()
    jacobian = dense_jacobian(reference)
    actions = [
        make_first_order_edge_action(
            lambda direction: jacobian @ direction,
            lambda upstream: jacobian.T @ upstream,
        ),
        make_first_order_edge_action(
            lambda direction: jacobian @ direction,
            lambda upstream: jacobian.T @ upstream,
        ),
    ]
    result = substitute_cached_messages(
        exact,
        {
            "active": True,
            "mode": 3,
            "clean_mask": torch.tensor([True, False]),
            "g_ref": references,
            "vectors": current,
            "vectors_ref": torch.stack((reference, reference)),
            "action_fns": actions,
        },
    )
    assert torch.allclose(result[0], references[0] + jacobian @ (current[0] - reference))
    assert torch.equal(result[1], exact[1])
