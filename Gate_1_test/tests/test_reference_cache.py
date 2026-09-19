import pytest
import numpy as np
import torch
from frost.e1.reference_cache import PerEdgeReferenceCache, EdgeCacheEntry
from frost.e1.mace_patch import substitute_cached_messages


def _edge(r_ij):
    r_ij = np.asarray(r_ij, dtype=float)
    distance = float(np.linalg.norm(r_ij))
    return {
        "r_ij": r_ij,
        "d_ij": distance,
        "rhat_ij": r_ij / distance,
    }

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


def test_substitute_cached_messages_mode_zero_ignores_clean_mask():
    mji = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    state = {
        "active": True,
        "mode": 0,
        "clean_mask": torch.empty(0, dtype=torch.bool),
        "g_ref": torch.empty(0),
    }

    result = substitute_cached_messages(mji, state)

    assert torch.equal(result, mji)


def test_insert_lookup_and_reuse_is_a_cache_hit():
    cache = PerEdgeReferenceCache([0.5])
    key = (0, 1, (0, 0, 0))
    edges = {key: _edge([1.0, 0.0, 0.0])}

    cache.initialize(edges)
    stats = cache.step(edges, timestep=1)

    assert stats["cache_hits"][0.5] == 1
    assert stats["cache_misses"][0.5] == 0
    assert cache._entries[key].state == "VALID"


def test_dirty_edge_refreshes_and_increments_generation():
    cache = PerEdgeReferenceCache([0.5])
    key = (0, 1, (0, 0, 0))
    cache.initialize({key: _edge([1.0, 0.0, 0.0])})

    stats = cache.step({key: _edge([1.7, 0.0, 0.0])}, timestep=1)

    assert stats["cache_misses"][0.5] == 1
    assert stats["refresh_counts"][0.5] == 1
    assert cache._entries[key].generation == 1
    assert cache._entries[key].state == "VALID"


def test_removed_edge_is_not_reused_and_reappearing_edge_is_new():
    cache = PerEdgeReferenceCache([0.5])
    key = (0, 1, (0, 0, 0))
    cache.initialize({key: _edge([1.0, 0.0, 0.0])})

    removed = cache.step({}, timestep=1)
    assert removed["removed_keys"] == {key}
    assert key not in cache._entries

    reappeared = cache.step({key: _edge([1.0, 0.0, 0.0])}, timestep=2)
    assert reappeared["new_keys"] == {key}
    assert reappeared["dirty_keys_by_eps"][0.5] == {key}
    assert cache._entries[key].generation == 0


def test_neighbor_order_does_not_change_physical_cache_identity():
    first = {
        (0, 1, (0, 0, 0)): _edge([1.0, 0.0, 0.0]),
        (1, 0, (0, 0, 0)): _edge([-1.0, 0.0, 0.0]),
    }
    reordered = dict(reversed(list(first.items())))
    cache = PerEdgeReferenceCache([0.5])
    cache.initialize(first)
    stats = cache.step(reordered, timestep=1)

    assert stats["n_edges_new"] == 0
    assert stats["n_edges_removed"] == 0
    assert stats["cache_hits"][0.5] == 2


def test_tolerance_masks_share_the_pre_refresh_reference_snapshot():
    key = (0, 1, (0, 0, 0))
    cache = PerEdgeReferenceCache([0.5, 1.0])
    cache.initialize({key: _edge([1.0, 0.0, 0.0])})

    stats = cache.step({key: _edge([1.7, 0.0, 0.0])}, timestep=1)

    assert stats["dirty_counts"][0.5] == 1
    assert stats["dirty_counts"][1.0] == 0
    assert stats["cache_misses"][1.0] == 0
