"""
toposeed/visualize.py — Cortical sheet and selectivity visualisations for TopoSeedLayer.

All functions save PNGs to disk and close figures immediately (Agg backend safe).

Public API
----------
plot_cortical_sheet_weights(layer, out_path, title=None)
    Heat-map grid of the H×W weight sheet (one panel per input-feature slice,
    or a summary projection), with active-mask and patch-boundary overlays.

plot_active_mask(layer, out_path, title=None)
    Binary active/dormant map coloured by patch membership.

plot_selectivity_maps(layer, dataloader, class_names, device, out_path, title=None,
                      n_samples_per_class=200)
    Per-class selectivity maps on the cortical sheet.
    selectivity_c = mean_activation_class_c  −  mean_activation_other_classes

plot_activation_map(layer, out_path, title=None)
    Three-panel map showing, per neuron on the H×W sheet:
      • mean |activation| from the last training batch
      • gradient magnitude from the last backward pass
      • evidence EMA (grad × act running average that drives expansion)

plot_all(layer, layer_name, dataloader, class_names, device, out_dir,
         epoch=None, prefix="")
    Convenience wrapper: calls all four functions above.

plot_multi_layer_selectivity(topo_layers, layer_names, dataloader, class_names,
                              device, out_path, title=None, n_samples_per_class=200)
    Side-by-side selectivity comparison across multiple layers in one figure.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn

from .layer import TopoSeedLayer


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _collect_activations(
    layer: TopoSeedLayer,
    dataloader,
    class_names: Sequence[str],
    device: torch.device,
    n_samples_per_class: int,
) -> np.ndarray:
    """
    Run `dataloader` through a forward-hook on `layer` and collect per-class
    mean activations.

    Returns
    -------
    act_by_class : np.ndarray  shape (n_classes, n_out)
        Mean absolute activation per output neuron per class.
    """
    n_classes = len(class_names)
    n_out = layer.n_out

    sums = np.zeros((n_classes, n_out), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)

    captured: list[torch.Tensor] = []

    def _hook(mod, inp, out):
        if layer.layer_type == "linear":
            captured.append(out.detach().cpu())    # (B, n_out)
        else:
            captured.append(out.detach().cpu().mean(dim=(2, 3)))  # (B, n_out)

    handle = layer.register_forward_hook(_hook)

    # We need to run the model (not just the layer). Use the module's parent
    # model — caller is expected to pass the correct dataloader for the model.
    # We just reuse the already-registered forward hook; caller drives the loop.
    # So this function is driven externally via the public API below.
    handle.remove()

    return sums, counts   # placeholder; real collection in _collect_activations_external


def _collect_per_class_activations(
    model: nn.Module,
    layer: TopoSeedLayer,
    dataloader,
    class_names: Sequence[str],
    device: torch.device,
    n_samples_per_class: int,
) -> np.ndarray:
    """
    Drive `model` in eval mode and collect per-class mean activation at `layer`.

    Returns
    -------
    act_by_class : np.ndarray  shape (n_classes, n_out)
    """
    n_classes = len(class_names)
    n_out = layer.n_out

    sums = np.zeros((n_classes, n_out), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)

    captured: list[torch.Tensor] = []

    def _hook(mod, inp, out):
        if layer.layer_type == "linear":
            captured.append(out.detach().cpu())
        else:
            captured.append(out.detach().cpu().mean(dim=(2, 3)))

    handle = layer.register_forward_hook(_hook)
    model.eval()

    with torch.no_grad():
        for imgs, labels in dataloader:
            # Stop once we have enough samples per class
            if (counts >= n_samples_per_class).all():
                break

            imgs = imgs.to(device)
            captured.clear()
            _ = model(imgs)

            if not captured:
                continue
            acts = captured[0].numpy()   # (B, n_out)

            for b in range(acts.shape[0]):
                c = int(labels[b].item())
                if counts[c] >= n_samples_per_class:
                    continue
                sums[c] += acts[b]
                counts[c] += 1

    handle.remove()

    counts_safe = np.maximum(counts[:, None], 1)
    return (sums / counts_safe).astype(np.float32)


def _patch_boundary_overlay(ax, patch_id: np.ndarray, active: np.ndarray) -> None:
    """
    Draw thin lines around each patch region on axes `ax`.
    patch_id: (H, W) int array, -1 = dormant
    active:   (H, W) bool array
    """
    H, W = patch_id.shape
    # Horizontal edges
    for i in range(H - 1):
        for j in range(W):
            if patch_id[i, j] != patch_id[i + 1, j]:
                ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5],
                        color="white", lw=0.6, alpha=0.6)
    # Vertical edges
    for i in range(H):
        for j in range(W - 1):
            if patch_id[i, j] != patch_id[i, j + 1]:
                ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5],
                        color="white", lw=0.6, alpha=0.6)


def _mark_seeds(ax, seed_positions: list, active: np.ndarray) -> None:
    """Mark seed positions with a small cross."""
    for (i, j) in seed_positions:
        if active[i, j]:
            ax.plot(j, i, "w+", markersize=4, markeredgewidth=0.8, alpha=0.8)


# ---------------------------------------------------------------------------
# Public visualisation functions
# ---------------------------------------------------------------------------

def plot_cortical_sheet_weights(
    layer: TopoSeedLayer,
    out_path: str | Path,
    title: Optional[str] = None,
    max_slices: int = 16,
) -> None:
    """
    Visualise the H×W cortical sheet.

    Shows two panels:
    1. **L2-norm projection** — per-neuron L2 norm of its weight vector (D dims),
       giving a single-value map of representational strength.
    2. **Top-K input slices** — up to `max_slices` individual D-slices of the
       weight tensor (input features), laid out in a grid.

    Dormant neurons are shown in black. Active patch boundaries and seed
    positions are overlaid on both panels.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    H, W, D = layer.H, layer.W, layer.D
    weights = _to_numpy(layer.sheet.weights)           # (H, W, D)
    active = _to_numpy(layer.sheet.active_mask).astype(bool)  # (H, W)
    patch_id = _to_numpy(layer.sheet.patch_id.float())        # (H, W)

    # --- L2-norm panel -------------------------------------------------------
    l2 = np.linalg.norm(weights, axis=-1)              # (H, W)
    l2[~active] = np.nan                               # dormant → NaN → black

    # --- Input-slice panels --------------------------------------------------
    n_slices = min(max_slices, D)
    # Pick evenly spaced D-slices
    slice_indices = np.round(np.linspace(0, D - 1, n_slices)).astype(int)
    slices = [weights[:, :, d] for d in slice_indices]

    n_cols = int(math.ceil(math.sqrt(n_slices)))
    n_rows_slices = int(math.ceil(n_slices / n_cols))

    # Layout: slice grid on the left (n_cols columns), L2 heatmap on the right
    # (one extra column).  The L2 column is sized so the square-pixel heatmap
    # fills the full height of the slice grid.
    cell_w, cell_h = 2.2, 2.5
    slice_grid_h = n_rows_slices * cell_h
    # L2 column width: at aspect="equal" the image will be W/H × height_in
    l2_col_w = max(2.0, slice_grid_h * W / max(H, 1)) + 0.7  # +0.7 for colorbar
    fig_w = n_cols * cell_w + l2_col_w
    fig_h = slice_grid_h + 0.5   # 0.5 for suptitle

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    fig.patch.set_facecolor("#1a1a2e")

    # width_ratios: equal slice columns + one wider L2 column
    width_ratios = [cell_w] * n_cols + [l2_col_w]
    gs = fig.add_gridspec(
        n_rows_slices, n_cols + 1,
        width_ratios=width_ratios,
        hspace=0.35, wspace=0.18,
        left=0.01, right=0.99, top=0.93, bottom=0.01,
    )

    # Input-feature slices — left n_cols columns
    vabs = max(np.nanmax(np.abs(np.stack(slices))), 1e-8)
    cmap_w = plt.cm.RdBu_r.copy()
    cmap_w.set_bad("black")
    for idx, (d_idx, sl) in enumerate(zip(slice_indices, slices)):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        sl_masked = sl.copy()
        sl_masked[~active] = np.nan
        ax.imshow(sl_masked, cmap=cmap_w, vmin=-vabs, vmax=vabs,
                  aspect="equal", interpolation="nearest")
        ax.set_title(f"d={d_idx}", color="white", fontsize=6)
        ax.axis("off")

    # L2-norm heatmap — rightmost column, spanning all rows
    ax_l2 = fig.add_subplot(gs[:, -1])
    cmap_l2 = plt.cm.inferno.copy()
    cmap_l2.set_bad("black")
    im = ax_l2.imshow(l2, cmap=cmap_l2, aspect="equal", interpolation="nearest")
    plt.colorbar(im, ax=ax_l2, fraction=0.06, pad=0.03, shrink=0.6)
    _patch_boundary_overlay(ax_l2, patch_id.astype(int), active)
    _mark_seeds(ax_l2, layer.sheet.seed_positions, active)
    ax_l2.set_title("Weight L2-Norm\nper Neuron", color="white", fontsize=8)
    ax_l2.axis("off")

    suptitle = title or f"Cortical Sheet Weights  [{H}×{W}×{D}]"
    fig.suptitle(suptitle, color="white", fontsize=11, y=0.98)

    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_active_mask(
    layer: TopoSeedLayer,
    out_path: str | Path,
    title: Optional[str] = None,
) -> None:
    """
    Show the H×W active/dormant map, coloured by patch membership.
    Seeds are marked with crosses.  Dormant neurons are black.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    H, W = layer.H, layer.W
    active = _to_numpy(layer.sheet.active_mask).astype(bool)
    patch_id_np = _to_numpy(layer.sheet.patch_id.float()).astype(int)

    num_patches = len(layer.sheet.seed_positions)
    # Build a colour image: each patch gets a distinct colour, dormant = black
    cmap = plt.cm.get_cmap("tab20", max(num_patches, 1))
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for pid in range(num_patches):
        mask = (patch_id_np == pid) & active
        colour = cmap(pid % 20)[:3]
        rgb[mask] = colour

    fig, ax = plt.subplots(figsize=(max(4, W * 0.3), max(3, H * 0.3)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    _patch_boundary_overlay(ax, patch_id_np, active)
    _mark_seeds(ax, layer.sheet.seed_positions, active)

    # Stats annotation
    n_active = int(active.sum())
    n_total = layer.n_out
    pct = 100.0 * n_active / max(n_total, 1)
    ax.set_xlabel(f"Active: {n_active}/{n_total} ({pct:.1f}%)",
                  color="white", fontsize=8)
    ax.set_title(title or "Active Mask & Patch Membership",
                 color="white", fontsize=9)
    ax.axis("off")

    # Legend: one patch per colour (cap at 20 for readability)
    handles = []
    for pid in range(min(num_patches, 20)):
        colour = cmap(pid % 20)[:3]
        handles.append(mpatches.Patch(color=colour, label=f"Patch {pid}"))
    if handles:
        ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
                  fontsize=5, framealpha=0.3, labelcolor="white",
                  facecolor="#1a1a2e")

    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_activation_map(
    layer: TopoSeedLayer,
    out_path: str | Path,
    title: Optional[str] = None,
) -> None:
    """
    Three-panel activation diagnostic for one layer.

    Panels (left → right):
    1. **Mean |activation|** — `layer._last_activation` mapped onto H×W.
       Shows which neurons fired strongly during the most recent training batch.
    2. **Gradient magnitude** — `layer._last_grad_magnitude` mapped onto H×W.
       Shows which neurons received the strongest learning signal.
    3. **Evidence EMA** — `layer._evidence_buf.ema` (running EMA of
       grad × act).  This is the signal the ExpansionManager uses to decide
       when to expand a patch; bright regions are about to grow.

    Dormant neurons are masked to black in all three panels.
    Patch boundaries and seed positions are overlaid.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    H, W = layer.H, layer.W
    active = _to_numpy(layer.sheet.active_mask).astype(bool)   # (H, W)
    patch_id_np = _to_numpy(layer.sheet.patch_id.float()).astype(int)

    def _masked(tensor_or_none) -> np.ndarray:
        """Convert (H, W) tensor to numpy, NaN-masking dormant cells."""
        if tensor_or_none is None:
            return np.full((H, W), np.nan)
        arr = _to_numpy(tensor_or_none)   # already (H, W) from layer hooks
        out = arr.copy()
        out[~active] = np.nan
        return out

    act_map  = _masked(layer._last_activation)
    grad_map = _masked(layer._last_grad_magnitude)
    # Evidence EMA lives in _evidence_buf.ema (H, W)
    ema_map  = _to_numpy(layer._evidence_buf.ema).copy()
    ema_map[~active] = np.nan

    panels = [
        (act_map,  plt.cm.hot,    "Mean |Activation|"),
        (grad_map, plt.cm.plasma, "Gradient Magnitude"),
        (ema_map,  plt.cm.viridis, "Evidence EMA\n(grad × act)"),
    ]

    cell = 2.8
    fig, axes = plt.subplots(1, 3, figsize=(cell * 3 + 0.8, cell + 0.7),
                              squeeze=True)
    fig.patch.set_facecolor("#1a1a2e")

    for ax, (data, cmap, label) in zip(axes, panels):
        cmap = cmap.copy()
        cmap.set_bad("black")
        im = ax.imshow(data, cmap=cmap, aspect="equal", interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        _patch_boundary_overlay(ax, patch_id_np, active)
        _mark_seeds(ax, layer.sheet.seed_positions, active)
        ax.set_title(label, color="white", fontsize=8)
        ax.axis("off")

    # Annotate active count
    n_active = int(active.sum())
    n_total  = layer.n_out
    fig.text(0.5, 0.01, f"Active: {n_active}/{n_total}",
             ha="center", color="white", fontsize=7)

    suptitle = title or f"Activation Diagnostics [{H}\u00d7{W} sheet]"
    fig.suptitle(suptitle, color="white", fontsize=10, y=1.01)

    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_selectivity_maps(
    model: nn.Module,
    layer: TopoSeedLayer,
    out_path: str | Path,
    dataloader,
    class_names: Sequence[str],
    device: torch.device,
    title: Optional[str] = None,
    n_samples_per_class: int = 200,
) -> None:
    """
    Compute per-class selectivity on the cortical sheet and save a figure.

    selectivity_c[i,j] = mean_act_c[i,j]  −  mean over all other classes

    One row per class, laid out as an H×W heat-map.  Dormant neurons are masked.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)
    # Collect activations
    act = _collect_per_class_activations(
        model, layer, dataloader, class_names, device, n_samples_per_class
    )  # (n_classes, n_out)

    H, W = layer.H, layer.W
    n_out = layer.n_out
    active = _to_numpy(layer.sheet.active_mask).astype(bool)   # (H, W)
    patch_id_np = _to_numpy(layer.sheet.patch_id.float()).astype(int)

    # selectivity: subtract mean of other classes
    total_mean = act.mean(axis=0)       # (n_out,)
    sel = act - total_mean[None, :]     # (n_classes, n_out)

    # Reshape to (n_classes, H, W)
    full = np.zeros((n_classes, H * W), dtype=np.float32)
    full[:, :n_out] = sel
    maps = full.reshape(n_classes, H, W)
    maps[:, ~active] = np.nan          # mask dormant neurons

    vabs = float(np.nanmax(np.abs(maps))) or 1.0
    cmap = plt.cm.RdGy_r.copy()
    cmap.set_bad("black")

    ncols = min(5, n_classes)
    nrows = math.ceil(n_classes / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.0 + 0.6),
                              squeeze=False)
    fig.patch.set_facecolor("#1a1a2e")

    for c in range(n_classes):
        row, col = divmod(c, ncols)
        ax = axes[row][col]
        im = ax.imshow(maps[c], cmap=cmap, vmin=-vabs, vmax=vabs,
                       aspect="auto", interpolation="nearest")
        _patch_boundary_overlay(ax, patch_id_np, active)
        _mark_seeds(ax, layer.sheet.seed_positions, active)
        ax.set_title(class_names[c], color="white", fontsize=8)
        ax.axis("off")

    # Hide empty panels
    for c in range(n_classes, nrows * ncols):
        row, col = divmod(c, ncols)
        axes[row][col].axis("off")

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cbar.ax.tick_params(colors="white", labelsize=7)
    cbar.set_label("selectivity", color="white", fontsize=8)

    suptitle = title or f"Category Selectivity [{H}×{W} sheet]"
    fig.suptitle(suptitle, color="white", fontsize=11)

    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_multi_layer_selectivity(
    model: nn.Module,
    topo_layers: List[TopoSeedLayer],
    layer_names: List[str],
    out_path: str | Path,
    dataloader,
    class_names: Sequence[str],
    device: torch.device,
    title: Optional[str] = None,
    n_samples_per_class: int = 200,
) -> None:
    """
    Side-by-side selectivity maps for multiple TopoSeed layers in one figure.
    Rows = classes, columns = layers.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)
    n_layers = len(topo_layers)

    # Collect activations for every layer simultaneously with one pass
    all_acts: list[np.ndarray | None] = [None] * n_layers
    sums_list = [np.zeros((n_classes, l.n_out), dtype=np.float64) for l in topo_layers]
    counts = np.zeros(n_classes, dtype=np.int64)
    captured_list: list[list[torch.Tensor]] = [[] for _ in topo_layers]

    handles = []
    for idx, layer in enumerate(topo_layers):
        def _make_hook(i):
            def _hook(mod, inp, out):
                if topo_layers[i].layer_type == "linear":
                    captured_list[i].append(out.detach().cpu())
                else:
                    captured_list[i].append(out.detach().cpu().mean(dim=(2, 3)))
            return _hook
        handles.append(layer.register_forward_hook(_make_hook(idx)))

    model.eval()
    with torch.no_grad():
        for imgs, labels in dataloader:
            if (counts >= n_samples_per_class).all():
                break
            for c_list in captured_list:
                c_list.clear()
            _ = model(imgs.to(device))

            if not all(len(c) > 0 for c in captured_list):
                continue

            for b in range(imgs.shape[0]):
                c = int(labels[b].item())
                if counts[c] >= n_samples_per_class:
                    continue
                for i, layer in enumerate(topo_layers):
                    sums_list[i][c] += captured_list[i][0][b].numpy()
                counts[c] += 1

    for h in handles:
        h.remove()

    counts_safe = np.maximum(counts[:, None], 1)
    # Build selectivity maps for each layer
    all_maps: list[np.ndarray] = []
    for i, layer in enumerate(topo_layers):
        act = (sums_list[i] / counts_safe).astype(np.float32)
        H, W, n_out = layer.H, layer.W, layer.n_out
        total_mean = act.mean(axis=0)
        sel = act - total_mean[None, :]
        active = _to_numpy(layer.sheet.active_mask).astype(bool)
        full = np.zeros((n_classes, H * W), dtype=np.float32)
        full[:, :n_out] = sel
        maps = full.reshape(n_classes, H, W)
        maps[:, ~active] = np.nan
        all_maps.append(maps)

    # vabs across all layers
    vabs = max(float(np.nanmax(np.abs(np.concatenate(
        [m.reshape(-1) for m in all_maps], axis=0
    )))), 1e-8)

    cmap = plt.cm.RdGy_r.copy()
    cmap.set_bad("black")

    fig, axes = plt.subplots(
        n_classes, n_layers,
        figsize=(n_layers * 2.2, n_classes * 1.8 + 0.8),
        squeeze=False,
    )
    fig.patch.set_facecolor("#1a1a2e")

    for c in range(n_classes):
        for li, (layer, maps) in enumerate(zip(topo_layers, all_maps)):
            ax = axes[c][li]
            active = _to_numpy(layer.sheet.active_mask).astype(bool)
            patch_id_np = _to_numpy(layer.sheet.patch_id.float()).astype(int)
            im = ax.imshow(maps[c], cmap=cmap, vmin=-vabs, vmax=vabs,
                           aspect="auto", interpolation="nearest")
            _patch_boundary_overlay(ax, patch_id_np, active)
            _mark_seeds(ax, layer.sheet.seed_positions, active)
            ax.axis("off")
            if c == 0:
                ax.set_title(layer_names[li], color="white", fontsize=8)
            if li == 0:
                ax.set_ylabel(class_names[c], color="white", fontsize=7)
                ax.yaxis.set_label_position("left")

    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cbar.ax.tick_params(colors="white", labelsize=7)
    cbar.set_label("selectivity", color="white", fontsize=8)

    fig.suptitle(title or "Multi-Layer Category Selectivity",
                 color="white", fontsize=11)
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience batch function
# ---------------------------------------------------------------------------

def plot_all(
    model: nn.Module,
    layer: TopoSeedLayer,
    layer_name: str,
    dataloader,
    class_names: Sequence[str],
    device: torch.device,
    out_dir: str | Path,
    epoch: Optional[int] = None,
    prefix: str = "",
) -> None:
    """
    Save all four visualisation types for one layer into typed subdirectories.

    For epoch-wise periodic saves the files are written under:
        {out_dir}/debug/epoch{epoch:03d}/weights/
        {out_dir}/debug/epoch{epoch:03d}/masks/
        {out_dir}/debug/epoch{epoch:03d}/selectivities/
        {out_dir}/debug/epoch{epoch:03d}/activations/

    For final saves (epoch=None) the files go to:
        {out_dir}/weights/
        {out_dir}/masks/
        {out_dir}/selectivities/
        {out_dir}/activations/

    Files:
        {prefix}{layer_name}_weights[_ep{epoch}].png
        {prefix}{layer_name}_mask[_ep{epoch}].png
        {prefix}{layer_name}_selectivity[_ep{epoch}].png
        {prefix}{layer_name}_activations[_ep{epoch}].png
    """
    out_dir = Path(out_dir)
    suffix = f"_ep{epoch:03d}" if epoch is not None else ""

    if epoch is not None:
        base = out_dir / "debug" / f"epoch{epoch:03d}"
    else:
        base = out_dir

    weights_dir       = base / "weights"
    masks_dir         = base / "masks"
    selectivities_dir = base / "selectivities"
    activations_dir   = base / "activations"

    plot_cortical_sheet_weights(
        layer,
        weights_dir / f"{prefix}{layer_name}_weights{suffix}.png",
        title=f"{layer_name} — Cortical Sheet Weights" + (f" (ep {epoch})" if epoch else ""),
    )
    plot_active_mask(
        layer,
        masks_dir / f"{prefix}{layer_name}_mask{suffix}.png",
        title=f"{layer_name} — Active Mask" + (f" (ep {epoch})" if epoch else ""),
    )
    plot_selectivity_maps(
        model, layer,
        selectivities_dir / f"{prefix}{layer_name}_selectivity{suffix}.png",
        dataloader, class_names, device,
        title=f"{layer_name} — Selectivity" + (f" (ep {epoch})" if epoch else ""),
    )
    plot_activation_map(
        layer,
        activations_dir / f"{prefix}{layer_name}_activations{suffix}.png",
        title=f"{layer_name} — Activation Diagnostics" + (f" (ep {epoch})" if epoch else ""),
    )
