import pytest
import numpy as np
from frost.e1.reference_cache import PerEdgeReferenceCache, EdgeCacheEntry

def test_cache_initialization():
    cache = PerEdgeReferenceCache(tolerances=[0.1, 1.0])
    
    edges_t0 = {
        (0, 1, (0,0,0)): {
            "r_ij": np.array([1.0, 0.0, 0.0]),
            "d_ij": 1.0,
            "rhat_ij": np.array([1.0, 0.0, 0.0])
        }
    }
    
    cache.initialize(edges_t0, timestep=0)
    assert cache.n_edges == 1
    assert cache._entries[(0,1,(0,0,0))].age == 0

def test_cache_step():
    cache = PerEdgeReferenceCache(tolerances=[0.5])
    
    edges_t0 = {
        (0, 1, (0,0,0)): {
            "r_ij": np.array([1.0, 0.0, 0.0]),
            "d_ij": 1.0,
            "rhat_ij": np.array([1.0, 0.0, 0.0])
        }
    }
    cache.initialize(edges_t0, timestep=0)
    
    # Step 1: Small movement (not dirty)
    edges_t1 = {
        (0, 1, (0,0,0)): {
            "r_ij": np.array([1.1, 0.0, 0.0]),
            "d_ij": 1.1,
            "rhat_ij": np.array([1.0, 0.0, 0.0])
        }
    }
    stats1 = cache.step(edges_t1, timestep=1)
    
    assert stats1["dirty_counts"][0.5] == 0
    assert cache._entries[(0,1,(0,0,0))].age == 1
    
    # Step 2: Large movement (dirty)
    edges_t2 = {
        (0, 1, (0,0,0)): {
            "r_ij": np.array([1.6, 0.0, 0.0]),
            "d_ij": 1.6,
            "rhat_ij": np.array([1.0, 0.0, 0.0])
        }
    }
    stats2 = cache.step(edges_t2, timestep=2)
    assert stats2["dirty_counts"][0.5] == 1
    assert cache._entries[(0,1,(0,0,0))].age == 0 # Reset!
    assert cache._entries[(0,1,(0,0,0))].lifetime_history == [1]

def test_new_and_removed_edges():
    cache = PerEdgeReferenceCache(tolerances=[0.5])
    
    edges_t0 = {
        (0, 1, (0,0,0)): {
            "r_ij": np.array([1.0, 0.0, 0.0]),
            "d_ij": 1.0,
            "rhat_ij": np.array([1.0, 0.0, 0.0])
        }
    }
    cache.initialize(edges_t0, timestep=0)
    
    # Remove (0,1), add (1,2)
    edges_t1 = {
        (1, 2, (0,0,0)): {
            "r_ij": np.array([0.0, 1.0, 0.0]),
            "d_ij": 1.0,
            "rhat_ij": np.array([0.0, 1.0, 0.0])
        }
    }
    stats1 = cache.step(edges_t1, timestep=1)
    
    assert stats1["n_edges_new"] == 1
    assert stats1["n_edges_removed"] == 1
    assert stats1["dirty_counts"][0.5] == 1 # New edge is dirty
    assert (0,1,(0,0,0)) not in cache._entries
    assert (1,2,(0,0,0)) in cache._entries
