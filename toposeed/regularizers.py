"""
toposeed/regularizers.py — Intra-layer and cross-layer topographic regularizers.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from .sheet import CorticalSheet


# ---------------------------------------------------------------------------
# Intra-layer smoothness
# ---------------------------------------------------------------------------

def intra_layer_smoothness(
    sheet: CorticalSheet,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Within each patch, penalise representational dissimilarity between
    neighbouring active neurons.

    For each active neuron, compute cosine similarity with its 4-connected
    active neighbours that belong to the same patch. Penalise low similarity.

    Returns a scalar tensor (the loss term).
    """
    H, W, D = sheet.H, sheet.W, sheet.D
    weights = sheet.weights  # (H, W, D), attached to autograd

    # We build shift-based neighbour comparisons (vectorised over the sheet).
    # Shifts: up, down, left, right
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    loss = weights.new_zeros(1)
    pair_count = 0

    for di, dj in shifts:
        # Shift the weight sheet
        shifted_w = torch.roll(weights, shifts=(-di, -dj), dims=(0, 1))  # (H, W, D)
        shifted_mask = torch.roll(active_mask, shifts=(-di, -dj), dims=(0, 1))
        shifted_pid = torch.roll(sheet.patch_id, shifts=(-di, -dj), dims=(0, 1))

        # Valid pairs: both active and same patch
        valid = (active_mask > 0) & (shifted_mask > 0) & \
                (sheet.patch_id == shifted_pid) & (sheet.patch_id >= 0)

        # Boundary correction: rolled-in values from the other side are invalid
        if di == -1:
            valid[-1, :] = False
        elif di == 1:
            valid[0, :] = False
        if dj == -1:
            valid[:, -1] = False
        elif dj == 1:
            valid[:, 0] = False

        if not valid.any():
            continue

        # Cosine similarity between neighbours
        w = weights[valid]            # (N, D)
        sw = shifted_w[valid]         # (N, D)
        cos_sim = F.cosine_similarity(w, sw, dim=1)  # (N,)

        # Penalise dissimilarity (1 - cos_sim) ∈ [0, 2]
        loss = loss + (1.0 - cos_sim).mean()
        pair_count += 1

    if pair_count > 0:
        loss = loss / pair_count
    return loss.squeeze()


# ---------------------------------------------------------------------------
# Cross-layer coherence
# ---------------------------------------------------------------------------

def cross_layer_coherence(
    sheet_L: CorticalSheet,
    sheet_L1: CorticalSheet,
    weight_matrix: torch.Tensor,
    num_pairs: int = 512,
) -> torch.Tensor:
    """
    Penalise spatial incoherence between adjacent layers L and L+1.

    Nearby neurons on sheet_L should have similar outgoing connection patterns
    to sheet_L+1.

    Steps:
    1. Collect active neuron flat indices on sheet_L.
    2. Sample `num_pairs` random pairs.
    3. For each pair (a, b):
       - spatial_dist  = Euclidean distance on the H×W sheet
       - connection_sim = cosine similarity of their rows in weight_matrix
       - contribution  = spatial_dist_normalised * connection_sim
         (nearby neurons having divergent connections → high loss)
    4. Return mean contribution as scalar loss.

    weight_matrix: (n_out_L, n_in_L) — the weight matrix of layer L.
    Note: n_out_L == n_neurons on sheet_L (active ones).
    """
    H_L, W_L = sheet_L.H, sheet_L.W
    active_L = sheet_L.active_mask.nonzero(as_tuple=False)   # (N_active, 2)
    N_active = active_L.shape[0]

    if N_active < 2:
        return weight_matrix.new_zeros(1).squeeze()

    # Limit pairs to what we have
    num_pairs = min(num_pairs, N_active * (N_active - 1) // 2)
    if num_pairs == 0:
        return weight_matrix.new_zeros(1).squeeze()

    # Normaliser: maximum possible distance on the sheet
    max_dist = float((H_L ** 2 + W_L ** 2) ** 0.5) + 1e-8

    # Random pair indices
    idx_a = torch.randint(0, N_active, (num_pairs,), device=sheet_L.weights.device)
    idx_b = torch.randint(0, N_active, (num_pairs,), device=sheet_L.weights.device)
    # Avoid self-pairs
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % N_active

    # Positions of sampled neurons
    pos_a = active_L[idx_a].float()   # (num_pairs, 2)
    pos_b = active_L[idx_b].float()   # (num_pairs, 2)
    spatial_dist = (pos_a - pos_b).pow(2).sum(-1).sqrt() / max_dist  # (num_pairs,)

    # Flat indices into weight_matrix rows
    # Map (i,j) → flat index in n_neurons space for sheet_L
    # We use i * W_L + j as proxy (valid because weights_to_matrix uses reshape)
    flat_a = (active_L[idx_a, 0] * W_L + active_L[idx_a, 1]).clamp(
        max=weight_matrix.shape[0] - 1)
    flat_b = (active_L[idx_b, 0] * W_L + active_L[idx_b, 1]).clamp(
        max=weight_matrix.shape[0] - 1)

    rows_a = weight_matrix[flat_a]   # (num_pairs, n_in)
    rows_b = weight_matrix[flat_b]   # (num_pairs, n_in)
    conn_sim = F.cosine_similarity(rows_a, rows_b, dim=1)  # (num_pairs,)

    # Nearby neurons should have SIMILAR connections → penalise nearby pairs
    # that are dissimilar.  Loss = spatial_proximity * dissimilarity
    proximity = 1.0 - spatial_dist   # high = close
    loss = (proximity * (1.0 - conn_sim)).mean()
    return loss
