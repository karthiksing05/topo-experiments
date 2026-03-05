"""
toposeed/sheet.py — CorticalSheet: spatial state management for one TopoSeed layer.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn

from .utils import (
    compute_sheet_dimensions,
    place_seeds_on_grid,
    correlated_init,
    active_mask_expanded,
    weights_to_matrix,
)


class CorticalSheet(nn.Module):
    """
    Manages all spatial state for one layer's cortical sheet.

    Parameters
    ----------
    H, W : spatial dimensions (H * W >= n_neurons)
    D    : depth = in_features (or c_in * k * k for conv)
    grid_size : G  →  G×G seeds at initialisation
    n_neurons : actual number of output neurons (may be < H*W)
    """

    def __init__(
        self,
        H: int,
        W: int,
        D: int,
        grid_size: int,
        n_neurons: int,
        std: float = 1.0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.D = D
        self.G = grid_size
        self.n_neurons = n_neurons

        # ---- weight tensor (H, W, D) ----------------------------------------
        seed_positions = place_seeds_on_grid(H, W, grid_size)
        grid_spacing = max(H / grid_size, W / grid_size)
        init_weights = correlated_init(H, W, D, seed_positions, grid_spacing,
                                       std=std, device=device)
        self.weights = nn.Parameter(init_weights)

        # ---- active mask (H, W) — non-parameter buffer ----------------------
        active = torch.zeros(H, W, dtype=torch.float32,
                             device=device if device is not None else torch.device("cpu"))
        for (i, j) in seed_positions:
            active[i, j] = 1.0
        self.register_buffer("active_mask", active)

        # ---- patch membership (H, W) soft assignment ------------------------
        # For v1 each neuron belongs to exactly one patch (hard assignment).
        # patch_id[i,j] in {0,..,G*G-1} for active neurons, -1 otherwise.
        num_patches = grid_size * grid_size
        patch_id = torch.full((H, W), -1, dtype=torch.long,
                               device=device if device is not None else torch.device("cpu"))
        for pid, (i, j) in enumerate(seed_positions):
            patch_id[i, j] = pid
        self.register_buffer("patch_id", patch_id)

        # confidence score per neuron (H, W), used for update_membership
        confidence = torch.zeros(H, W, dtype=torch.float32,
                                  device=device if device is not None else torch.device("cpu"))
        for (i, j) in seed_positions:
            confidence[i, j] = 1.0
        self.register_buffer("confidence", confidence)

        # seed_positions: list[tuple[int,int]], one per patch (indexed by pid)
        self.seed_positions: List[Tuple[int, int]] = seed_positions

        # track total expansions / deaths for diagnostics
        self.total_expansions: int = 0
        self.total_deaths: int = 0
        self.expansions_this_epoch: int = 0
        self.deaths_this_epoch: int = 0

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

    def get_active_weight_matrix(self) -> torch.Tensor:
        """
        Return weight matrix (n_neurons, D) with dormant neurons zeroed out,
        preserving autograd through the mask multiplication.
        """
        mask = active_mask_expanded(self.active_mask, self.D)  # (H, W, D)
        masked = self.weights * mask                            # autograd-safe
        n_out = self.n_neurons
        return masked.reshape(self.H * self.W, self.D)[:n_out, :]  # (n_out, D)

    # ------------------------------------------------------------------
    # Spatial helpers (no-grad, called from expansion manager)
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        i: int,
        j: int,
        radius: int = 1,
    ) -> List[Tuple[int, int]]:
        """
        Return (i, j) positions within Manhattan distance `radius`
        that are currently dormant (active_mask == 0) and within the sheet.
        """
        result: List[Tuple[int, int]] = []
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.H and 0 <= nj < self.W:
                    if self.active_mask[ni, nj].item() == 0.0:
                        result.append((ni, nj))
        return result

    # ------------------------------------------------------------------
    # Neuron lifecycle (called inside torch.no_grad())
    # ------------------------------------------------------------------

    def activate_neuron(
        self,
        i: int,
        j: int,
        pid: int,
        source_weights: torch.Tensor,
        gradient: torch.Tensor | None,
    ) -> None:
        """
        Wake a dormant neuron at (i,j):
        - Set active_mask[i,j] = 1
        - Initialise weights as source_weights + small perturbation biased toward gradient
        - Assign to patch `pid`
        """
        with torch.no_grad():
            perturb = torch.randn_like(source_weights) * 0.01
            if gradient is not None and gradient.shape == source_weights.shape:
                # Bias the perturbation in the gradient direction (grow in direction of signal)
                gnorm = gradient.norm()
                if gnorm > 1e-8:
                    perturb = perturb + gradient / gnorm * 0.005
            self.weights.data[i, j] = source_weights + perturb
            self.active_mask[i, j] = 1.0
            self.patch_id[i, j] = pid
            self.confidence[i, j] = 0.5          # initial confidence
        self.total_expansions += 1
        self.expansions_this_epoch += 1

    def deactivate_neuron(self, i: int, j: int) -> None:
        """
        Kill active neuron at (i,j): zero its weights, mark dormant.
        """
        with torch.no_grad():
            self.weights.data[i, j] = 0.0
            self.active_mask[i, j] = 0.0
            self.patch_id[i, j] = -1
            self.confidence[i, j] = 0.0
        self.total_deaths += 1
        self.deaths_this_epoch += 1

    # ------------------------------------------------------------------
    # Membership update
    # ------------------------------------------------------------------

    def update_membership(self, beta: float = 0.7) -> None:
        """
        Recompute patch_id for neurons whose confidence is below `beta`
        by reassigning them to the closest seed (in spatial distance).
        High-confidence neurons stay put (winner-takes-all).
        """
        with torch.no_grad():
            active_coords = self.active_mask.nonzero(as_tuple=False)  # (N_active, 2)
            if active_coords.numel() == 0:
                return

            # Seed positions as tensor (num_patches, 2)
            seeds = torch.tensor(
                self.seed_positions,
                dtype=torch.float32,
                device=self.weights.device,
            )

            for idx in range(active_coords.shape[0]):
                i = int(active_coords[idx, 0].item())
                j = int(active_coords[idx, 1].item())
                conf = self.confidence[i, j].item()
                if conf < beta:
                    # Reassign to nearest seed (Euclidean on sheet)
                    pos = torch.tensor(
                        [i, j], dtype=torch.float32, device=self.weights.device
                    )
                    dists = (seeds - pos).pow(2).sum(-1)
                    nearest_pid = int(dists.argmin().item())
                    self.patch_id[i, j] = nearest_pid

    # ------------------------------------------------------------------
    # Epoch bookkeeping
    # ------------------------------------------------------------------

    def reset_epoch_counters(self) -> None:
        self.expansions_this_epoch = 0
        self.deaths_this_epoch = 0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def active_count(self) -> int:
        return int(self.active_mask.sum().item())

    def dormant_count(self) -> int:
        return self.n_neurons - self.active_count()

    def num_patches(self) -> int:
        return len(self.seed_positions)

    def mean_patch_size(self) -> float:
        n = self.num_patches()
        if n == 0:
            return 0.0
        return self.active_count() / n
