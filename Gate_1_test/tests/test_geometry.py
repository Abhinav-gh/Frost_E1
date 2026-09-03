import pytest
import numpy as np
from ase import Atoms
from frost.e1.geometry import (
    minimum_image_displacement,
    compute_edge_geometry,
    build_edge_geometry_from_ase,
    delta_r,
    delta_rhat,
    is_dirty_zeroth_order,
    compute_all_perturbations_batch
)

def test_minimum_image_displacement():
    cell = np.eye(3) * 10.0
    # atoms close to opposite boundaries
    pos_i = np.array([0.1, 5.0, 5.0])
    pos_j = np.array([9.9, 5.0, 5.0])
    
    # direct displacement
    diff = pos_j - pos_i
    assert np.allclose(diff, [9.8, 0, 0])
    
    # with MIC shift (-1 along x)
    shift = np.array([-1, 0, 0])
    r_ij = minimum_image_displacement(pos_i, pos_j, cell, shift)
    assert np.allclose(r_ij, [-0.2, 0, 0])

def test_compute_edge_geometry():
    r_ij = np.array([3.0, 4.0, 0.0])
    d_ij, rhat_ij = compute_edge_geometry(r_ij)
    assert np.isclose(d_ij, 5.0)
    assert np.allclose(rhat_ij, [3/5, 4/5, 0.0])

def test_build_edge_geometry_from_ase():
    atoms = Atoms('H2', positions=[[0,0,0], [1.0, 0, 0]], cell=[10,10,10], pbc=True)
    edges = build_edge_geometry_from_ase(atoms, cutoff=2.0)
    
    # should have (0,1) and (1,0)
    assert (0, 1, (0,0,0)) in edges
    assert (1, 0, (0,0,0)) in edges
    
    e = edges[(0, 1, (0,0,0))]
    assert np.isclose(e['d_ij'], 1.0)
    assert np.allclose(e['rhat_ij'], [1.0, 0.0, 0.0])

def test_delta_functions():
    r1 = np.array([1.0, 0.0, 0.0])
    r2 = np.array([1.1, 0.0, 0.0])
    assert np.isclose(delta_r(r2, r1), 0.1)
    
    # rhat
    rhat1 = np.array([1.0, 0.0, 0.0])
    rhat2 = np.array([0.0, 1.0, 0.0])
    assert np.isclose(delta_rhat(rhat2, rhat1), np.sqrt(2))

def test_is_dirty_zeroth_order():
    d_ref = 2.0
    rhat_ref = np.array([1.0, 0.0, 0.0])
    
    # Same pos
    assert not is_dirty_zeroth_order(d_ref, rhat_ref, d_ref, rhat_ref, epsilon=0.1)
    
    # moved slightly
    d_cur = 2.05
    assert not is_dirty_zeroth_order(d_cur, rhat_ref, d_ref, rhat_ref, epsilon=0.1)
    assert is_dirty_zeroth_order(d_cur, rhat_ref, d_ref, rhat_ref, epsilon=0.04)
