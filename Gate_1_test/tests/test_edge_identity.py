import torch

from frost.e1.geometry import compare_edge_key_sets, mace_edge_keys


def test_mace_keys_preserve_direction_and_periodic_offset():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 1]])
    unit_shifts = torch.tensor([
        [0, 0, 0],
        [0, 0, 0],
        [-1, 0, 0],
    ], dtype=torch.float64)

    keys = mace_edge_keys(edge_index, unit_shifts)

    assert keys == [
        (0, 1, (0, 0, 0)),
        (1, 0, (0, 0, 0)),
        (0, 1, (-1, 0, 0)),
    ]


def test_reordered_mace_edges_still_match_by_key():
    ase_keys = [
        (0, 1, (0, 0, 0)),
        (1, 0, (0, 0, 0)),
        (0, 1, (-1, 0, 0)),
    ]
    edge_index = torch.tensor([[0, 0, 1], [1, 1, 0]])
    unit_shifts = torch.tensor([
        [0, 0, 0],
        [-1, 0, 0],
        [0, 0, 0],
    ], dtype=torch.float64)

    mace_keys = mace_edge_keys(edge_index, unit_shifts)
    report = compare_edge_key_sets(ase_keys, mace_keys)

    assert report["matched"]
    assert report["missing_from_mace"] == []
    assert report["missing_from_ase"] == []
    assert report["mace_duplicates"] == []