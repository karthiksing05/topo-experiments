"""
toposeed/layer.py — TopoSeedLayer: drop-in replacement for nn.Linear / nn.Conv2d.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sheet import CorticalSheet
from .buffers import EvidenceBuffer, HealthBuffer, ResidualBuffer
from .expansion import ExpansionManager
from .regularizers import intra_layer_smoothness
from .utils import compute_sheet_dimensions


class TopoSeedLayer(nn.Module):
    """
    Drop-in replacement for nn.Linear or nn.Conv2d with topographic organisation.

    The layer manages:
    - A CorticalSheet (the learnable weight tensor reshaped onto a 2D sheet)
    - An ExpansionManager (evidence → expansion → death lifecycle)
    - Regularisation losses (intra-layer smoothness)

    Usage
    -----
    # Replace nn.Linear(784, 256):
    layer = TopoSeedLayer(layer_type='linear', in_features=784, out_features=256)

    # Replace nn.Conv2d(64, 128, 3, padding=1):
    layer = TopoSeedLayer(layer_type='conv', in_channels=64, out_channels=128,
                          kernel_size=3, padding=1)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        layer_type: str,                  # 'linear' or 'conv'
        # Linear params
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
        # Conv params
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        kernel_size: Optional[int] = None,
        stride: int = 1,
        padding: int = 0,
        # TopoSeed params
        grid_size: int = 4,
        warmup_steps: int = 500,
        expansion_threshold: float = 0.15,
        death_threshold: float = 0.02,
        death_sustained_steps: int = 300,
        expansion_radius: int = 1,
        residual_weight: float = 0.5,
        beta: float = 0.7,
        lambda_intra: float = 0.01,
    ):
        super().__init__()

        # ---- validate and store layer type --------------------------------
        assert layer_type in ("linear", "conv"), \
            "layer_type must be 'linear' or 'conv'"
        self.layer_type = layer_type
        self.lambda_intra = lambda_intra

        # ---- derive dimensions --------------------------------------------
        if layer_type == "linear":
            assert in_features is not None and out_features is not None
            self.in_features = in_features
            self.out_features = out_features
            n_out = out_features
            D = in_features                # depth = in_features
            self.stride = 1
            self.padding = 0
            self.kernel_size = None
            # Kaiming uniform std
            std = math.sqrt(2.0 / in_features)
        else:  # conv
            assert in_channels is not None and out_channels is not None
            assert kernel_size is not None
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            n_out = out_channels
            D = in_channels * kernel_size * kernel_size
            std = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))

        # ---- cortical sheet dimensions ------------------------------------
        H, W = compute_sheet_dimensions(n_out)
        self.H = H
        self.W = W
        self.D = D
        self.n_out = n_out

        # ---- build the cortical sheet -------------------------------------
        self.sheet = CorticalSheet(
            H=H, W=W, D=D,
            grid_size=grid_size,
            n_neurons=n_out,
            std=std,
        )

        # ---- bias (standard nn.Parameter) ---------------------------------
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_out))
        else:
            self.register_parameter("bias", None)

        # ---- buffers ----------------------------------------------------------
        num_patches = grid_size * grid_size
        self._evidence_buf = EvidenceBuffer(H, W)
        self._health_buf = HealthBuffer(H, W)
        self._residual_buf = ResidualBuffer(num_patches)

        # ---- expansion manager --------------------------------------------
        self._expansion_mgr = ExpansionManager(
            sheet=self.sheet,
            evidence_buffer=self._evidence_buf,
            health_buffer=self._health_buf,
            residual_buffer=self._residual_buf,
            warmup_steps=warmup_steps,
            expansion_threshold=expansion_threshold,
            death_threshold=death_threshold,
            death_sustained_steps=death_sustained_steps,
            expansion_radius=expansion_radius,
            residual_weight=residual_weight,
            beta=beta,
        )
        self.warmup_steps = warmup_steps

        # ---- internal state captured by hooks -----------------------------
        self._last_activation: Optional[torch.Tensor] = None   # (H, W)
        self._last_grad_magnitude: Optional[torch.Tensor] = None  # (H, W)
        self._reg_loss: torch.Tensor = torch.zeros(1)

        # Register hooks
        self._fwd_hook = self.register_forward_hook(self._forward_hook)
        self._bwd_hook = self.register_full_backward_hook(self._backward_hook)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get masked weight matrix (n_out, D) — autograd-safe via multiplication
        W_masked = self.sheet.get_active_weight_matrix()   # (n_out, D)

        if self.layer_type == "linear":
            out = F.linear(x, W_masked, self.bias)
        else:
            # Reshape to conv weight tensor (out_channels, in_channels, kH, kW)
            W_conv = W_masked.view(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            )
            out = F.conv2d(x, W_conv, self.bias,
                           stride=self.stride, padding=self.padding)

        # Compute and store intra-layer regularisation loss
        self._reg_loss = (
            intra_layer_smoothness(self.sheet, self.sheet.active_mask) * self.lambda_intra
        )
        return out

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _forward_hook(
        self,
        module: "TopoSeedLayer",
        input: tuple,
        output: torch.Tensor,
    ) -> None:
        """Capture mean absolute activation per output neuron."""
        with torch.no_grad():
            if self.layer_type == "linear":
                # output: (batch, n_out)
                act_flat = output.detach().abs().mean(0)  # (n_out,)
            else:
                # output: (batch, out_channels, H_out, W_out)
                act_flat = output.detach().abs().mean(dim=(0, 2, 3))  # (out_channels,)

            # Map flat (n_out,) back onto the (H, W) sheet
            n_out = self.n_out
            padded = torch.zeros(self.H * self.W, device=act_flat.device)
            padded[:n_out] = act_flat
            self._last_activation = padded.view(self.H, self.W)

    def _backward_hook(
        self,
        module: "TopoSeedLayer",
        grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        """Capture per-neuron gradient magnitude from grad_output."""
        with torch.no_grad():
            if len(grad_output) == 0 or grad_output[0] is None:
                return
            go = grad_output[0].detach()

            if self.layer_type == "linear":
                # go: (batch, n_out) — L2 norm across batch
                grad_flat = go.pow(2).mean(0).sqrt()   # (n_out,)
            else:
                # go: (batch, out_channels, H_out, W_out)
                grad_flat = go.pow(2).mean(dim=(0, 2, 3)).sqrt()  # (out_channels,)

            n_out = self.n_out
            padded = torch.zeros(self.H * self.W, device=grad_flat.device)
            padded[:n_out] = grad_flat
            self._last_grad_magnitude = padded.view(self.H, self.W)

    # ------------------------------------------------------------------
    # Buffer update (called by training loop after loss.backward())
    # ------------------------------------------------------------------

    def update_buffers(
        self,
        patch_residuals: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Trigger expansion manager update.  Must be called AFTER loss.backward().
        If patch_residuals is None, compute a proxy from stored activations.
        """
        if self._last_grad_magnitude is None or self._last_activation is None:
            return

        gm = self._last_grad_magnitude.to(self.sheet.weights.device)
        act = self._last_activation.to(self.sheet.weights.device)

        if patch_residuals is None:
            patch_residuals = self._compute_patch_residuals(act)

        self._expansion_mgr.step_update(gm, act, patch_residuals)

    def _compute_patch_residuals(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Simple proxy residual per patch: mean activation of neurons NOT in that
        patch (how much signal a patch is leaving unexplained).
        """
        num_patches = len(self.sheet.seed_positions)
        residuals = torch.zeros(num_patches, device=self.sheet.weights.device)
        total_act = activation.sum()
        for pid in range(num_patches):
            in_patch = self.sheet.patch_id == pid
            patch_act = activation[in_patch].sum() if in_patch.any() else activation.new_zeros(1)
            residuals[pid] = (total_act - patch_act).clamp(min=0.0)
        # Normalise
        if total_act > 1e-8:
            residuals = residuals / (total_act + 1e-8)
        return residuals

    # ------------------------------------------------------------------
    # Accessors for training loop
    # ------------------------------------------------------------------

    def get_reg_loss(self) -> torch.Tensor:
        """Return the intra-layer regularisation loss for this step."""
        return self._reg_loss

    def get_stats(self) -> dict:
        """Return diagnostic stats dict."""
        active = self.sheet.active_count()
        dormant = self.sheet.dormant_count()
        mean_ev = float(self._evidence_buf.ema[self.sheet.active_mask > 0].mean().item()) \
            if active > 0 else 0.0
        return {
            "active_neuron_count": active,
            "dormant_neuron_count": dormant,
            "total_neurons": self.n_out,
            "num_patches": self.sheet.num_patches(),
            "mean_patch_size": self.sheet.mean_patch_size(),
            "mean_evidence": mean_ev,
            "expansions_this_epoch": self.sheet.expansions_this_epoch,
            "deaths_this_epoch": self.sheet.deaths_this_epoch,
            "warmup_step": self._expansion_mgr.current_step,
            "in_warmup": self._expansion_mgr.current_step < self.warmup_steps,
        }

    def reset_epoch_stats(self) -> None:
        self.sheet.reset_epoch_counters()

    # ------------------------------------------------------------------
    # Device handling
    # ------------------------------------------------------------------

    def _sync_non_module_buffers(self) -> None:
        """Move the plain-tensor buffers (not nn.Module objects) to the
        same device as the layer parameters. Works for cuda, mps, and cpu."""
        device = next(self.parameters()).device
        self._evidence_buf = self._evidence_buf.to(device)
        self._health_buf = self._health_buf.to(device)
        self._residual_buf = self._residual_buf.to(device)

    def _apply(self, fn, recurse: bool = True):
        """Called by .to(), .cuda(), .mps(), .float(), etc. — including when
        a parent module calls .to() on the whole model tree.  This is the
        correct hook to sync non-parameter buffers after device/dtype moves."""
        result = super()._apply(fn, recurse=recurse)
        result._sync_non_module_buffers()
        return result

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        result._sync_non_module_buffers()
        return result

    def cuda(self, device=None):
        result = super().cuda(device)
        result._sync_non_module_buffers()
        return result

    def mps(self):
        """Move the layer to Apple Silicon MPS device."""
        result = super().to(torch.device("mps"))
        result._sync_non_module_buffers()
        return result

    # ------------------------------------------------------------------
    # Correctness checks (exposed as public methods)
    # ------------------------------------------------------------------

    def assert_dormant_zero(self, x: torch.Tensor) -> None:
        """
        Verify that dormant neurons contribute zero to the forward pass.
        Asserts that output is the same whether dormant weights are zero or not.
        """
        out_normal = self.forward(x)

        # Temporarily save dormant weights, set them to random nonzero
        with torch.no_grad():
            dormant = self.sheet.active_mask == 0
            saved = self.sheet.weights.data[dormant].clone()
            self.sheet.weights.data[dormant] = torch.randn_like(saved)

        out_perturbed = self.forward(x)

        # Restore
        with torch.no_grad():
            self.sheet.weights.data[dormant] = saved

        diff = (out_normal - out_perturbed).abs().max().item()
        assert diff < 1e-5, (
            f"Dormant neurons contribute non-zero output! max diff={diff:.2e}"
        )

    def assert_seed_count(self) -> None:
        """
        Assert active_mask.sum() == grid_size^2 immediately after init.
        (Call once right after constructing the layer, before any training.)
        """
        expected = self._expansion_mgr.sheet.G ** 2
        actual = int(self.sheet.active_mask.sum().item())
        assert actual == expected, (
            f"Expected {expected} active seeds, got {actual}"
        )

    def assert_no_expansion_during_warmup(self) -> None:
        """
        Assert active_mask has not changed during warmup.
        (Relies on step counter; call within training loop as desired.)
        """
        if self._expansion_mgr.current_step < self.warmup_steps:
            expected = self.sheet.G ** 2
            actual = int(self.sheet.active_mask.sum().item())
            assert actual == expected, (
                f"Expansion occurred during warmup at step "
                f"{self._expansion_mgr.current_step}! active={actual}"
            )

    @staticmethod
    def assert_sheet_reshape_roundtrip(H: int, W: int, D: int) -> None:
        """Assert that weight matrix → cortical sheet → weight matrix is lossless."""
        original = torch.randn(H * W, D)
        sheet = original.reshape(H, W, D)
        recovered = sheet.reshape(H * W, D)
        assert torch.allclose(original, recovered), \
            "Sheet reshape roundtrip is not lossless!"
