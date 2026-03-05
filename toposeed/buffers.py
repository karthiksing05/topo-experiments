"""
toposeed/buffers.py — EvidenceBuffer, HealthBuffer, and ResidualBuffer.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


class EvidenceBuffer:
    """
    Tracks gradient_magnitude * activation for each neuron over a rolling
    window using an EMA. Used to decide when to expand a patch.
    """

    def __init__(
        self,
        H: int,
        W: int,
        window_size: int = 100,
        ema_alpha: float = 0.05,
        device: torch.device | None = None,
    ):
        self.H = H
        self.W = W
        self.ema_alpha = ema_alpha
        _dev = device if device is not None else torch.device("cpu")
        self.ema: torch.Tensor = torch.zeros(H, W, device=_dev)
        self.step_count: int = 0

    # ------------------------------------------------------------------

    def update(
        self,
        grad_magnitude: torch.Tensor,   # (H, W) – L2 norm of grad per neuron
        activation: torch.Tensor,        # (H, W) – mean |activation| per neuron
        active_mask: torch.Tensor,       # (H, W) – binary
    ) -> None:
        """
        signal = grad_magnitude * activation (element-wise, active neurons only)
        ema    = (1 - alpha) * ema + alpha * signal
        """
        signal = grad_magnitude * activation * active_mask
        self.ema = (1.0 - self.ema_alpha) * self.ema + self.ema_alpha * signal
        self.step_count += 1

    def get_patch_evidence(self, patch_id: torch.Tensor, pid: int) -> float:
        """Return mean EMA for all neurons belonging to patch `pid`."""
        mask = patch_id == pid
        if mask.sum() == 0:
            return 0.0
        return float(self.ema[mask].mean().item())

    def reset_patch(self, patch_id: torch.Tensor, pid: int) -> None:
        """Zero out EMA for neurons in patch `pid` after an expansion event."""
        # Use torch.where instead of boolean-indexed assignment for MPS compat.
        self.ema = torch.where(
            patch_id == pid,
            torch.zeros_like(self.ema),
            self.ema,
        )

    def to(self, device: torch.device) -> "EvidenceBuffer":
        self.ema = self.ema.to(device)
        return self


# ---------------------------------------------------------------------------

class HealthBuffer:
    """
    Slower rolling average for death eligibility.
    A neuron is eligible to die if its health score is consistently low.
    """

    MIN_SIGNAL_THRESHOLD: float = 1e-5   # below this → increment low_health_count

    def __init__(
        self,
        H: int,
        W: int,
        window_size: int = 500,
        ema_alpha: float = 0.01,
        device: torch.device | None = None,
    ):
        self.H = H
        self.W = W
        self.ema_alpha = ema_alpha
        _dev = device if device is not None else torch.device("cpu")
        self.health: torch.Tensor = torch.ones(H, W, device=_dev)
        self.low_health_count: torch.Tensor = torch.zeros(H, W, dtype=torch.long,
                                                           device=_dev)

    # ------------------------------------------------------------------

    def update(
        self,
        grad_magnitude: torch.Tensor,
        activation: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> None:
        signal = grad_magnitude * activation * active_mask
        self.health = (1.0 - self.ema_alpha) * self.health + self.ema_alpha * signal
        # Increment low_health_count for active neurons below signal threshold.
        # Use arithmetic ops instead of boolean-indexed assignment for MPS compat.
        low = ((signal < self.MIN_SIGNAL_THRESHOLD) & (active_mask > 0)).long()
        recovered = ((signal >= self.MIN_SIGNAL_THRESHOLD) & (active_mask > 0))
        self.low_health_count = self.low_health_count + low
        # Reset counter to 0 for neurons that recovered.
        self.low_health_count = torch.where(
            recovered,
            torch.zeros_like(self.low_health_count),
            self.low_health_count,
        )

    def get_death_candidates(
        self,
        threshold: float,
        sustained_steps: int,
    ) -> List[Tuple[int, int]]:
        """
        Return (i, j) positions where:
        - health < threshold  AND
        - low_health_count > sustained_steps
        """
        eligible = (self.health < threshold) & (self.low_health_count > sustained_steps)
        coords = eligible.nonzero(as_tuple=False)
        return [(int(r[0].item()), int(r[1].item())) for r in coords]

    def to(self, device: torch.device) -> "HealthBuffer":
        self.health = self.health.to(device)
        self.low_health_count = self.low_health_count.to(device)
        return self


# ---------------------------------------------------------------------------

class ResidualBuffer:
    """
    Tracks per-patch residual error via an EMA.
    Used as a complement to gradient magnitude for expansion decisions.
    """

    def __init__(
        self,
        num_patches: int,
        ema_alpha: float = 0.05,
        device: torch.device | None = None,
    ):
        self.num_patches = num_patches
        self.ema_alpha = ema_alpha
        _dev = device if device is not None else torch.device("cpu")
        self.residual_ema: torch.Tensor = torch.zeros(num_patches, device=_dev)

    def update(self, patch_residuals: torch.Tensor) -> None:
        """patch_residuals: shape (num_patches,) — scalar error per patch."""
        self.residual_ema = (
            (1.0 - self.ema_alpha) * self.residual_ema
            + self.ema_alpha * patch_residuals.to(self.residual_ema.device)
        )

    def get_patch_residual(self, pid: int) -> float:
        if pid < 0 or pid >= self.num_patches:
            return 0.0
        return float(self.residual_ema[pid].item())

    def to(self, device: torch.device) -> "ResidualBuffer":
        self.residual_ema = self.residual_ema.to(device)
        return self
