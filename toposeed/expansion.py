"""
toposeed/expansion.py — Expansion and death logic for one TopoSeed layer.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from .sheet import CorticalSheet
from .buffers import EvidenceBuffer, HealthBuffer, ResidualBuffer


class ExpansionManager:
    """
    Manages the evidence accrual → expansion → death lifecycle for one layer.
    """

    def __init__(
        self,
        sheet: CorticalSheet,
        evidence_buffer: EvidenceBuffer,
        health_buffer: HealthBuffer,
        residual_buffer: ResidualBuffer,
        warmup_steps: int,
        expansion_threshold: float,
        death_threshold: float,
        death_sustained_steps: int,
        expansion_radius: int = 1,
        residual_weight: float = 0.5,
        beta: float = 0.7,
    ):
        self.sheet = sheet
        self.evidence_buffer = evidence_buffer
        self.health_buffer = health_buffer
        self.residual_buffer = residual_buffer
        self.warmup_steps = warmup_steps
        self.expansion_threshold = expansion_threshold
        self.death_threshold = death_threshold
        self.death_sustained_steps = death_sustained_steps
        self.expansion_radius = expansion_radius
        self.residual_weight = residual_weight
        self.beta = beta
        self._step = 0

    # ------------------------------------------------------------------
    # Main entry point (called every training step after backward)
    # ------------------------------------------------------------------

    def step_update(
        self,
        grad_magnitude: torch.Tensor,    # (H, W)
        activation: torch.Tensor,         # (H, W)
        patch_residuals: torch.Tensor,    # (num_patches,)
    ) -> None:
        self._step += 1
        self.evidence_buffer.update(grad_magnitude, activation, self.sheet.active_mask)
        self.health_buffer.update(grad_magnitude, activation, self.sheet.active_mask)
        self.residual_buffer.update(patch_residuals)

        if self._step < self.warmup_steps:
            return

        self._check_expansions()
        self._check_deaths()
        self.sheet.update_membership(self.beta)

    # ------------------------------------------------------------------

    def _check_expansions(self) -> None:
        num_patches = len(self.sheet.seed_positions)
        for pid in range(num_patches):
            evidence = self.evidence_buffer.get_patch_evidence(self.sheet.patch_id, pid)
            residual = self.residual_buffer.get_patch_residual(pid)
            signal = evidence + self.residual_weight * residual

            if signal > self.expansion_threshold:
                self._expand_patch(pid)
                self.evidence_buffer.reset_patch(self.sheet.patch_id, pid)

    def _expand_patch(self, pid: int) -> None:
        boundary = self._get_patch_boundary(pid)
        for (i, j) in boundary:
            dormant_neighbors = self.sheet.get_neighbors(i, j, self.expansion_radius)
            for (ni, nj) in dormant_neighbors:
                source_weights = self.sheet.weights.data[i, j].clone()
                gradient = self._get_gradient_at(i, j)
                self.sheet.activate_neuron(ni, nj, pid, source_weights, gradient)

    def _check_deaths(self) -> None:
        candidates = self.health_buffer.get_death_candidates(
            self.death_threshold, self.death_sustained_steps
        )
        active_count = self.sheet.active_count()
        for (i, j) in candidates:
            # Never kill the very last neuron
            if active_count <= 1:
                break
            if self.sheet.active_mask[i, j].item() == 1.0:
                self.sheet.deactivate_neuron(i, j)
                active_count -= 1

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------

    def _get_patch_boundary(self, pid: int) -> List[Tuple[int, int]]:
        """
        Return active neurons in patch `pid` that have at least one dormant
        neighbour within distance 1 — the expansion frontier.
        """
        # Neurons belonging to this patch
        mask = (self.sheet.patch_id == pid) & (self.sheet.active_mask > 0)
        coords = mask.nonzero(as_tuple=False)
        boundary: List[Tuple[int, int]] = []
        for row in coords:
            i, j = int(row[0].item()), int(row[1].item())
            if len(self.sheet.get_neighbors(i, j, radius=1)) > 0:
                boundary.append((i, j))
        return boundary

    def _get_gradient_at(self, i: int, j: int) -> torch.Tensor | None:
        if self.sheet.weights.grad is not None:
            return self.sheet.weights.grad[i, j].clone()
        return torch.zeros(self.sheet.D, device=self.sheet.weights.device)

    # ------------------------------------------------------------------

    @property
    def current_step(self) -> int:
        return self._step
