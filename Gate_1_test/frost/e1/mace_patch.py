import contextlib
import types
from typing import Optional, Any, Tuple

import torch


def substitute_cached_messages(mji: torch.Tensor, state: dict) -> torch.Tensor:
    """Apply a Frost cache to layer-1 messages without changing coordinates."""
    if not state.get("active", False):
        return mji
    clean_mask = state.get("clean_mask")
    g_ref = state.get("g_ref")
    if clean_mask is None or clean_mask.shape != (mji.shape[0],):
        raise ValueError("frost_cache clean_mask does not match MACE edges")
    if g_ref is None or g_ref.shape != mji.shape:
        raise ValueError("frost_cache g_ref does not match mji")
    mode = state.get("mode", 0)
    if mode == 0:
        return mji
    if mode == 1:
        replacement = g_ref
    elif mode == 2:
        J_ref = state.get("J_ref")
        vectors = state.get("vectors")
        vectors_ref = state.get("vectors_ref")
        if J_ref is None or J_ref.shape != (mji.shape[0], mji.shape[1], 3):
            raise ValueError("frost_cache J_ref must have shape [edges, features, 3]")
        if vectors is None or vectors_ref is None:
            raise ValueError("First-order cache requires current and reference vectors")
        replacement = g_ref + torch.einsum("ea,eba->eb", vectors - vectors_ref, J_ref)
    else:
        raise ValueError(f"Unsupported Frost cache mode: {mode}")
    return torch.where(clean_mask[:, None], replacement, mji)


@contextlib.contextmanager
def capture_layer1_vectors(mace_model, state: dict):
    """Capture the vector tensor that produced MACE's edge attributes."""
    def capture(_module, inputs, _output):
        state["vectors"] = inputs[0]

    handle = mace_model.spherical_harmonics.register_forward_hook(capture)
    try:
        yield
    finally:
        handle.remove()

def create_patched_forward(original_forward, frost_cache_state):
    """
    Creates a patched forward method for the first MACE InteractionBlock.
    
    frost_cache_state is a dict that controls the cache injection:
    {
        "active": bool,
        "mode": int,            # 0 for exact, 1 for zeroth-order, 2 for first-order
        "clean_mask": Tensor,   # boolean mask of shape [n_edges]
        "g_ref": Tensor,        # shape [n_edges, F]
        "J_ref": Tensor,        # shape [n_edges, F, 3] (needed if mode=2)
        "vectors": Tensor,      # shape [n_edges, 3] current vectors
        "vectors_ref": Tensor,  # shape [n_edges, 3] reference vectors
    }
    """
    def patched_forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_real = lammps_natoms[0] if lammps_class is not None else None
        
        # Copied from original forward up to mji computation
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        if hasattr(self, "handle_lammps"):
            node_feats = self.handle_lammps(
                node_feats,
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
                first_layer=first_layer,
            )
        tp_weights = self.conv_tp_weights(edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
            
        mji = self.conv_tp(
            node_feats[edge_index[0]], edge_attrs, tp_weights
        )

        # --- FROST INTERCEPTION ---
        if frost_cache_state.get("active", False) and first_layer:
            mode = frost_cache_state["mode"]
            clean_mask = frost_cache_state["clean_mask"]
            g_ref = frost_cache_state["g_ref"]
            
            if mode == 0:
                frost_cache_state["mji_exact"] = mji
            mji = substitute_cached_messages(mji, frost_cache_state)

            frost_cache_state["edge_index"] = edge_index
        
        # Continue with original logic
        from mace.tools.scatter import scatter_sum
        message = scatter_sum(
            src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
        )
        if hasattr(self, "truncate_ghosts"):
            message = self.truncate_ghosts(message, n_real)
            node_attrs = self.truncate_ghosts(node_attrs, n_real)
            sc = self.truncate_ghosts(sc, n_real)
        
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )
        
    return patched_forward


@contextlib.contextmanager
def apply_frost_patch(mace_model, frost_cache_state):
    """
    Context manager that temporarily patches the first interaction block of a MACE model.
    """
    first_interaction = mace_model.interactions[0]
    if hasattr(first_interaction, "conv_fusion"):
        raise RuntimeError(
            "Frost interception requires the unfused first MACE interaction block"
        )
    original_forward = first_interaction.forward
    
    # We must bind the method to the instance
    patched = create_patched_forward(original_forward, frost_cache_state)
    first_interaction.forward = types.MethodType(patched, first_interaction)
    
    try:
        yield
    finally:
        first_interaction.forward = original_forward
