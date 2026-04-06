"""topo_quant_loss.py
====================
Three variants of TopoQuantLoss — a quantization-aware training regulariser
inspired by the same cosine-similarity principle as TopoLoss.

Variants
--------
TopoQuantLoss        — Hard (straight-through) uniform quantisation during training.
SoftTopoQuantLoss    — Temperature-annealed soft quantisation via softmax assignment.
CombinedTopoQuantLoss — Spatial TopoLoss + TopoQuantLoss applied simultaneously.

All three penalise:
    L = -cosine_similarity(W_flat, Quantise(W_flat))
which encourages weight distributions to cluster naturally around the discrete
levels of the target bit-width.

Usage
-----
    from topo_quant_loss import TopoQuantLoss, SoftTopoQuantLoss, CombinedTopoQuantLoss

    loss_fn = TopoQuantLoss(num_bits=4, tau=1.0)
    # inside training loop:
    q_loss  = loss_fn(model)
    total   = task_loss + q_loss
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Low-level quantisation helpers
# ---------------------------------------------------------------------------

def _nearest_bin_centers(w: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Return the nearest quantisation bin center for each element of w.

    Computed fully detached so we get stable targets for the commitment loss.
    """
    n_levels = 2 ** num_bits - 1
    w_min  = w.detach().min()
    w_max  = w.detach().max()
    scale  = (w_max - w_min).clamp(min=1e-8) / n_levels
    q = torch.round((w.detach() - w_min) / scale) * scale + w_min
    return q


def _commitment_loss(w: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Commitment loss: penalise the distance of each weight from its nearest bin.

    Loss = MSE(w, nearest_bin_center(w)) / Var(w)
    Gradient = 2 * (w - bin_center) / Var(w)

    Dividing by Var(w) makes the loss scale-invariant across layers with
    different weight magnitudes, so tau has a consistent meaning regardless
    of layer depth or initialisation scale.

    This is analogous to the commitment term in VQ-VAE and is far more effective
    than a cosine-STE loss for the hard-quantisation variant because:
      1. The gradient is direct and proportional to how far each weight
         is from a bin, rather than depending on angular alignment.
      2. It does not suffer from the double-gradient-path instability of
         cosine_loss(w, STE(w)), where w appears in both arguments.
      3. cos(w, q(w)) for 4-bit weights is ~0.998 (quantisation barely shifts
         weight direction), making the cosine loss signal negligibly small.

    No STE is required since this is a standalone regulariser, not a
    forward-pass weight substitution.
    """
    q    = _nearest_bin_centers(w, num_bits)   # fully detached target
    var  = w.detach().var().clamp(min=1e-8)
    return F.mse_loss(w, q) / var


def _uniform_quantise(w: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Hard quantisation used only for post-training evaluation (evaluate_quantized).

    Returns the hard-rounded values; not used in any training loss.
    """
    n_levels = 2 ** num_bits - 1
    w_min  = w.detach().min()
    w_max  = w.detach().max()
    scale  = (w_max - w_min).clamp(min=1e-8) / n_levels
    return torch.round((w - w_min) / scale) * scale + w_min


def _soft_quantise(
    w: torch.Tensor,
    num_bits: int,
    temperature: float,
) -> torch.Tensor:
    """Soft quantisation via softmax assignment to nearest discrete levels.

    q_k are uniformly spaced in [w.min(), w.max()].
    Assignment probability: p_k = softmax(-|w - q_k|^2 / temperature)
    Output: sum_k p_k * q_k  (differentiable everywhere)
    """
    n_levels = 2 ** num_bits
    w_min = w.min().detach()
    w_max = w.max().detach()
    levels = torch.linspace(w_min.item(), w_max.item(), n_levels, device=w.device, dtype=w.dtype)

    # w:       [N]        after flatten
    # levels:  [L]
    w_flat  = w.reshape(-1, 1)           # [N, 1]
    lev     = levels.unsqueeze(0)        # [1, L]
    dists   = -(w_flat - lev) ** 2 / max(temperature, 1e-8)
    probs   = F.softmax(dists, dim=-1)   # [N, L]
    w_soft  = (probs * lev).sum(dim=-1)  # [N]
    return w_soft.reshape(w.shape)


def _cosine_loss(w: torch.Tensor, w_q: torch.Tensor) -> torch.Tensor:
    """1 - cosine_similarity(flatten(w), flatten(w_q))."""
    w_f  = w.reshape(1, -1)
    wq_f = w_q.reshape(1, -1)
    # Avoid division-by-zero for zero-weight layers at init
    cos = F.cosine_similarity(w_f, wq_f, dim=1, eps=1e-8)
    return (1.0 - cos).mean()


# ---------------------------------------------------------------------------
# Layer selection helper
# ---------------------------------------------------------------------------

def _weight_tensors(
    model: nn.Module,
    apply_to_layers: Optional[list],
    exclude_bias: bool,
):
    """Yield (name, weight_tensor) for layers selected by apply_to_layers.

    apply_to_layers=None  → all nn.Linear and nn.Conv2d weights.
    apply_to_layers=[str] → only parameters whose name contains one of the
                            provided substrings.
    """
    for name, param in model.named_parameters():
        if exclude_bias and name.endswith(".bias"):
            continue
        if apply_to_layers is None:
            # Default: Linear and Conv2d weight tensors only
            module_name = name.rsplit(".", 1)[0]
            try:
                mod = model.get_submodule(module_name)
            except AttributeError:
                continue
            if not isinstance(mod, (nn.Linear, nn.Conv2d)):
                continue
        else:
            if not any(sub in name for sub in apply_to_layers):
                continue
        yield name, param


# ---------------------------------------------------------------------------
# Variant 1 – Hard (STE) quantisation
# ---------------------------------------------------------------------------

class TopoQuantLoss(nn.Module):
    """Hard-quantisation commitment loss.

    Penalises each weight's distance from its nearest quantisation bin center:

        L = tau * mean_over_layers( MSE(W, nearest_bin(W)) )

    The gradient d(L)/dw = 2*(w - bin_center) acts as a restoring force
    pulling each weight toward a bin.  This is the commitment loss from VQ-VAE,
    adapted for quantisation-aware training without STE instability.

    Parameters
    ----------
    num_bits : int
        Target quantisation bit-width (2–8 recommended).
    tau : float
        Loss scaling factor.
    apply_to_layers : list[str] | None
        Layer name substrings to target (None = all Linear/Conv2d).
    exclude_bias : bool
        Whether to skip bias parameters.
    """

    def __init__(
        self,
        num_bits: int = 4,
        tau: float = 1.0,
        apply_to_layers: Optional[list] = None,
        exclude_bias: bool = True,
    ):
        super().__init__()
        self.num_bits        = num_bits
        self.tau             = tau
        self.apply_to_layers = apply_to_layers
        self.exclude_bias    = exclude_bias

    def forward(self, model: nn.Module) -> torch.Tensor:
        losses = []
        for _name, w in _weight_tensors(model, self.apply_to_layers, self.exclude_bias):
            losses.append(_commitment_loss(w, self.num_bits))
        if not losses:
            return torch.tensor(0.0, requires_grad=True)
        return self.tau * torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Variant 2 – Soft (temperature-annealed) quantisation
# ---------------------------------------------------------------------------

class SoftTopoQuantLoss(nn.Module):
    """Temperature-annealed soft quantisation.

    Parameters
    ----------
    num_bits : int
        Target bit-width.
    tau : float
        Loss scaling factor.
    initial_temperature : float
        Starting softmax temperature (high → uniform, low → hard argmax).
    final_temperature : float
        Ending temperature after annealing.
    anneal_steps : int
        Number of optimiser steps over which to anneal.
    anneal_schedule : str
        'cosine', 'linear', or 'exponential'.
    apply_to_layers : list[str] | None
        Layer substrings to target.
    exclude_bias : bool
        Skip bias parameters.
    """

    def __init__(
        self,
        num_bits: int = 4,
        tau: float = 1.0,
        initial_temperature: float = 5.0,
        final_temperature: float = 0.1,
        anneal_steps: int = 10_000,
        anneal_schedule: str = "cosine",
        apply_to_layers: Optional[list] = None,
        exclude_bias: bool = True,
    ):
        super().__init__()
        self.num_bits            = num_bits
        self.tau                 = tau
        self.T0                  = initial_temperature
        self.T1                  = final_temperature
        self.anneal_steps        = anneal_steps
        self.anneal_schedule     = anneal_schedule
        self.apply_to_layers     = apply_to_layers
        self.exclude_bias        = exclude_bias

    def _temperature(self, current_step: int) -> float:
        t = min(current_step / max(self.anneal_steps, 1), 1.0)
        if self.anneal_schedule == "linear":
            return self.T0 + (self.T1 - self.T0) * t
        elif self.anneal_schedule == "exponential":
            ratio = math.log(self.T1 / max(self.T0, 1e-8))
            return self.T0 * math.exp(ratio * t)
        else:  # cosine (default)
            return self.T1 + 0.5 * (self.T0 - self.T1) * (1.0 + math.cos(math.pi * t))

    def forward(self, model: nn.Module, current_step: int = 0) -> torch.Tensor:
        temperature = self._temperature(current_step)
        losses = []
        for _name, w in _weight_tensors(model, self.apply_to_layers, self.exclude_bias):
            w_q = _soft_quantise(w, self.num_bits, temperature)
            losses.append(_cosine_loss(w, w_q))
        if not losses:
            return torch.tensor(0.0, requires_grad=True)
        return self.tau * torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Variant 3 – Combined spatial topo + quantisation loss
# ---------------------------------------------------------------------------

class CombinedTopoQuantLoss(nn.Module):
    """Spatial TopoLoss + TopoQuantLoss applied simultaneously.

    Requires the `topoloss` package (used by the rest of this repo).

    Parameters
    ----------
    num_bits : int
        Target bit-width for quantisation.
    tau_spatial : float
        Scaling weight for the spatial TopoLoss term.
    tau_precision : float
        Scaling weight for the quantisation loss term.
    spatial_downsample_factor : int
        Downsampling factor passed to LaplacianPyramid.
    use_soft_quant : bool
        If True, use soft quantisation (SoftTopoQuantLoss internals).
    soft_temperature : float
        Temperature for soft quantisation (only used if use_soft_quant=True).
    apply_to_layers : list[str] | None
        Layer substrings to apply quantisation loss to.
    exclude_bias : bool
        Skip bias parameters.
    """

    def __init__(
        self,
        num_bits: int = 4,
        tau_spatial: float = 1.0,
        tau_precision: float = 1.0,
        spatial_downsample_factor: int = 3,
        use_soft_quant: bool = False,
        soft_temperature: float = 1.0,
        apply_to_layers: Optional[list] = None,
        exclude_bias: bool = True,
    ):
        super().__init__()
        self.num_bits                   = num_bits
        self.tau_spatial                = tau_spatial
        self.tau_precision              = tau_precision
        self.spatial_downsample_factor  = spatial_downsample_factor
        self.use_soft_quant             = use_soft_quant
        self.soft_temperature           = soft_temperature
        self.apply_to_layers            = apply_to_layers
        self.exclude_bias               = exclude_bias

        # find_cortical_sheet_size from topoloss tells us the 2D sheet dimensions
        try:
            from topoloss.core import find_cortical_sheet_size
            self._find_cortical_sheet_size = find_cortical_sheet_size
            self._topoloss_available       = True
        except ImportError:
            self._topoloss_available = False

    def _quant_loss(self, model: nn.Module, current_step: int = 0) -> torch.Tensor:
        losses = []
        for _name, w in _weight_tensors(model, self.apply_to_layers, self.exclude_bias):
            if self.use_soft_quant:
                # Soft variant: cosine loss against differentiable soft targets
                w_q = _soft_quantise(w, self.num_bits, self.soft_temperature)
                losses.append(_cosine_loss(w.detach(), w_q))
            else:
                # Hard variant: commitment loss toward nearest bin centers
                losses.append(_commitment_loss(w, self.num_bits))
        if not losses:
            return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
        return torch.stack(losses).mean()

    def _spatial_loss(self, model: nn.Module) -> torch.Tensor:
        """Spatial smoothness penalty on the cortical sheet.

        For each qualifying weight matrix, we build a 2-D cortical map of
        per-unit weight L2 norms, then compare it to a spatially blurred
        version via cosine similarity.  The blur is implemented as
        avg_pool2d → bilinear upsample, equivalent to the Laplacian pyramid
        downsampling step in topoloss.  This needs only PyTorch — topoloss is
        used only for find_cortical_sheet_size.
        """
        if not self._topoloss_available:
            return torch.tensor(0.0)
        factor = self.spatial_downsample_factor
        losses = []
        for name, param in model.named_parameters():
            if self.apply_to_layers is not None:
                if not any(sub in name for sub in self.apply_to_layers):
                    continue
            if param.ndim < 2:
                continue
            # Build scalar cortical map: per-unit L2 norm
            w_flat  = param.reshape(param.shape[0], -1)  # [n_units, fan_in]
            n_units = w_flat.shape[0]
            size    = self._find_cortical_sheet_size(n_units)
            h, w_   = size.height, size.width
            if h * w_ != n_units:
                continue
            unit_norms = w_flat.norm(dim=1)              # [n_units]
            sheet      = unit_norms.reshape(1, 1, h, w_) # [1, 1, h, w]
            # Blur: downsample then upsample (equivalent to LaplacianPyramid step)
            ph      = max(1, h // factor)
            pw      = max(1, w_ // factor)
            blurred = F.interpolate(
                F.adaptive_avg_pool2d(sheet, (ph, pw)),
                size=(h, w_),
                mode="bilinear",
                align_corners=False,
            )
            cos = F.cosine_similarity(
                sheet.reshape(1, -1),
                blurred.reshape(1, -1),
                dim=1, eps=1e-8,
            )
            losses.append((1.0 - cos).mean())
        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()

    def forward(self, model: nn.Module, current_step: int = 0) -> torch.Tensor:
        q_loss   = self._quant_loss(model, current_step)
        sp_loss  = self._spatial_loss(model)
        return self.tau_precision * q_loss + self.tau_spatial * sp_loss
