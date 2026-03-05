"""
toposeed/utils.py — Grid initialization, spatial helpers, cortical sheet utilities.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device(preferred: str = "auto") -> torch.device:
    """
    Return the best available torch.device.

    preferred:
        "auto"  – cuda > mps > cpu  (recommended)
        anything else is passed directly to torch.device()
    """
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Sheet/grid geometry
# ---------------------------------------------------------------------------

def compute_sheet_dimensions(n_neurons: int) -> Tuple[int, int]:
    """
    Given a number of output neurons, find H and W such that:
    - H * W >= n_neurons
    - H and W are as close to each other as possible (near-square)
    - H * W is minimized subject to the above

    Returns (H, W) with H <= W.
    """
    H = math.isqrt(n_neurons)
    # Walk upward until H * ceil(n_neurons / H) is minimised
    best_H, best_W = H, math.ceil(n_neurons / max(H, 1))
    best_area = best_H * best_W
    for h in range(max(1, H - 2), H + 3):
        w = math.ceil(n_neurons / h)
        area = h * w
        if area >= n_neurons and area < best_area:
            best_area = area
            best_H, best_W = h, w
    # Ensure H <= W
    if best_H > best_W:
        best_H, best_W = best_W, best_H
    return best_H, best_W


def place_seeds_on_grid(H: int, W: int, G: int) -> List[Tuple[int, int]]:
    """
    Place G×G seeds evenly on an H×W cortical sheet.
    Seeds are centered within each grid cell.
    Returns list of (i, j) positions of length G*G.
    """
    positions: List[Tuple[int, int]] = []
    for gi in range(G):
        for gj in range(G):
            # Centre of cell (gi, gj)
            i = int((gi + 0.5) * H / G)
            j = int((gj + 0.5) * W / G)
            # Clamp to valid range
            i = min(i, H - 1)
            j = min(j, W - 1)
            positions.append((i, j))
    return positions


def spatial_distance_matrix(positions: List[Tuple[int, int]]) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances between a list of (i,j) positions.
    Returns tensor of shape (N, N).
    """
    coords = torch.tensor(positions, dtype=torch.float32)  # (N, 2)
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)        # (N, N, 2)
    return torch.sqrt((diff ** 2).sum(-1))                  # (N, N)


# ---------------------------------------------------------------------------
# Spatially correlated initialization
# ---------------------------------------------------------------------------

def gaussian_blur_2d(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Apply 2D Gaussian blur to a tensor of shape (H, W, D).
    The blur is applied independently over the D dimension.
    sigma: standard deviation in pixels.
    """
    H, W, D = tensor.shape
    # Build a 1-D Gaussian kernel
    radius = max(int(math.ceil(3 * sigma)), 1)
    kernel_size = 2 * radius + 1
    x = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device) - radius
    gauss = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss = gauss / gauss.sum()

    # Reshape to (1, D, H, W) for grouped F.conv2d
    # tensor: (H, W, D) → (D, H, W) → (1, D, H, W)
    t = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, D, H, W)

    # Separable convolution: rows then cols.
    # groups=D: each of the D channels is convolved independently.
    # Weight shape for grouped conv: (C_out=D, C_in/groups=1, kH, kW)
    row_k = gauss.view(1, 1, 1, kernel_size).expand(D, 1, 1, kernel_size).contiguous()
    col_k = gauss.view(1, 1, kernel_size, 1).expand(D, 1, kernel_size, 1).contiguous()

    padW = (kernel_size - 1) // 2
    t = F.conv2d(t, row_k, padding=(0, padW), groups=D)   # (1, D, H, W)
    t = F.conv2d(t, col_k, padding=(padW, 0), groups=D)   # (1, D, H, W)

    # Back to (H, W, D)
    return t.squeeze(0).permute(1, 2, 0)


def correlated_init(
    H: int,
    W: int,
    D: int,
    seed_positions: List[Tuple[int, int]],
    grid_spacing: float,
    std: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build a (H, W, D) weight tensor with spatially correlated initialization
    for seed positions. Non-seed positions are zero.

    shared_noise = gaussian_blur(randn(H, W, D), sigma=grid_spacing * 0.5)
    independent_noise = randn(H, W, D) * 0.3
    seed_weights = kaiming_base + shared_noise * 0.2 + independent_noise * 0.8

    `std` is the Kaiming-uniform scale factor (computed externally).
    """
    if device is None:
        device = torch.device("cpu")

    raw = torch.randn(H, W, D, device=device) * std
    shared = gaussian_blur_2d(torch.randn(H, W, D, device=device) * std,
                               sigma=max(grid_spacing * 0.5, 0.5))
    independent = torch.randn(H, W, D, device=device) * std * 0.3

    weights = raw + shared * 0.2 + independent * 0.8
    return weights


# ---------------------------------------------------------------------------
# Mask / reshape helpers
# ---------------------------------------------------------------------------

def active_mask_expanded(mask: torch.Tensor, D: int) -> torch.Tensor:
    """
    Expand a (H, W) binary mask to (H, W, D) for element-wise weight masking.
    """
    return mask.unsqueeze(-1).expand(-1, -1, D)


def weights_to_matrix(weights: torch.Tensor, n_out: int) -> torch.Tensor:
    """
    Flatten the cortical sheet (H, W, D) back to a weight matrix (n_out, D).
    Only the first n_out rows of the flattened H*W dimension are used
    (the sheet may be slightly larger than n_out due to near-square rounding).
    """
    H, W, D = weights.shape
    return weights.reshape(H * W, D)[:n_out, :]


def matrix_to_sheet(
    matrix: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    """
    Pad or trim a weight matrix (n_out, D) to exactly H*W rows, then
    reshape to (H, W, D).
    """
    n_out, D = matrix.shape
    target = H * W
    if n_out < target:
        pad = torch.zeros(target - n_out, D, dtype=matrix.dtype, device=matrix.device)
        matrix = torch.cat([matrix, pad], dim=0)
    else:
        matrix = matrix[:target, :]
    return matrix.reshape(H, W, D)
