"""
Topography implemented with the same sparsity constraints as project with Zekun!

Two main constraints:
*   A batch-wide KL constraint to ensure the space gets used properly
*   A per-instance entropy constraint to ensure individual instances have low entropy

Both constraints are applied only on the downsampled cortical sheet of fc1 activations —
the same layer that TopoLoss is applied to — to ensure that when scaling back up, sparsity
is retained per-cluster and not per-neuron.

The model is a two-layer MLP (SimpleNN) trained on Fashion-MNIST (28×28 grayscale, 10 classes),
mirroring the architecture in the topoloss_demo notebook.  TopoLoss + KL/entropy sparsity
are applied exclusively to fc1.  A plain CE baseline and a topo-only (no sparsity) variant
are also trained for comparison.
"""

# -- Imports -------------------------------------------------------------------

import argparse
import copy
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from topoloss import LaplacianPyramid, TopoLoss
from topoloss.core import find_cortical_sheet_size

# -- Defaults ------------------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs" / "train_topo_sparsity"

FMNIST_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",  "Shirt",   "Sneaker",  "Bag",   "AnkleBoot",
]

# All layers captured for visualisation
VIS_LAYER_NAMES  = ["fc1", "fc2"]
# Layers that get TopoLoss + KL + entropy (fc2 left unconstrained)
TOPO_LAYER_NAMES = ["fc1"]


# -- Model ---------------------------------------------------------------------

class SimpleNN(nn.Module):
    """Two-layer MLP matching the demo notebook architecture."""
    def __init__(self, hidden_size: int = 256, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(28 * 28, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, 10, bias=bias)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x                        # logits


# -- Cortical-sheet activation regularizers ------------------------------------

def cortical_sparsity_losses(activations: torch.Tensor,
                             factor_h: float, factor_w: float,
                             temperature: float = 3.0) -> tuple:
    """KL-from-uniform + per-sample entropy on the downsampled cortical sheet.

    Conv activations (B,C,H,W) are globally avg-pooled to (B,C) first so that
    output channels form the topographic axes.
    """

    if activations.ndim == 4:
        activations = activations.mean(dim=(2, 3))

    B, N  = activations.shape
    size  = find_cortical_sheet_size(N)
    H, W  = size.height, size.width

    sheet      = activations.reshape(B, 1, H, W)
    H_down     = max(1, round(H / factor_h))
    W_down     = max(1, round(W / factor_w))
    flat       = F.adaptive_avg_pool2d(sheet, (H_down, W_down)).reshape(B, -1)
    M          = flat.shape[1]

    probs       = F.softmax(flat / temperature, dim=-1)
    batch_mean  = probs.mean(dim=0)
    kl_loss     = F.kl_div(
        torch.full_like(batch_mean, -math.log(M)),
        batch_mean, reduction="sum",
    )
    entropy_loss = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
    return kl_loss, entropy_loss


# -- Cortical-sheet debug visualisation ----------------------------------------

def save_debug_cortical_sheets(
    models: dict,
    val_loader: DataLoader,
    layer_cfg: dict,
    out_dir: Path,
    device: str,
) -> None:
    """For each model save a debug figure showing per-layer cortical sheets.

    One PNG per model saved to out_dir/debug_cortical_sheets_{label}.png.
    Each figure: rows = up to 8 val samples, column groups per TOPO layer:
      raw sheet | downsampled | softmax probs | activation histogram

    Also prints NaN/inf statistics and KL/entropy values to stdout.
    """
    print("\n" + "=" * 65)
    print("  CORTICAL SHEET DEBUG")
    print("=" * 65)

    n_show = 8

    def _collect_one_batch(model):
        """Return {layer_name: activation_tensor (B,...)} for the first batch."""
        store   = {n: None for n in TOPO_LAYER_NAMES}
        handles = []
        for name in TOPO_LAYER_NAMES:
            layer = getattr(model, name)
            def _hook(mod, inp, out, n=name):
                store[n] = out.detach().cpu()
            handles.append(layer.register_forward_hook(_hook))
        model.eval()
        with torch.no_grad():
            imgs, _ = next(iter(val_loader))
            model(imgs.to(device))
        for h in handles:
            h.remove()
        return store, imgs.cpu()

    def _sheet_decompose(act1d, lname):
        """(N,) → raw H×W, downsampled H_d×W_d, softmax probs H_d×W_d)."""
        lc   = layer_cfg[lname]
        N    = act1d.shape[0]
        size = find_cortical_sheet_size(N)
        H, W = size.height, size.width
        raw  = act1d[:H * W].reshape(H, W)
        H_d  = max(1, round(H / lc["factor_h"]))
        W_d  = max(1, round(W / lc["factor_w"]))
        down  = F.adaptive_avg_pool2d(
            raw.unsqueeze(0).unsqueeze(0), (H_d, W_d)
        ).squeeze()
        probs = down.flatten().softmax(0).reshape(H_d, W_d)
        return raw, down, probs

    def _print_stats(t, label):
        valid = t[~t.isnan() & ~t.isinf()]
        nans  = t.isnan().sum().item()
        infs  = t.isinf().sum().item()
        if valid.numel():
            print(f"    {label:22s} "
                  f"min={valid.min():.4g}  max={valid.max():.4g}  "
                  f"mean={valid.mean():.4g}  NaN={nans}  Inf={infs}")
        else:
            print(f"    {label:22s} ALL NaN/Inf  NaN={nans}  Inf={infs}")

    for label, model in models.items():
        print(f"\n  Model: {label}")
        batch_acts, batch_imgs = _collect_one_batch(model)
        B = min(n_show, batch_imgs.shape[0])

        # print stats
        for name in TOPO_LAYER_NAMES:
            raw_act = batch_acts[name][:B]
            if raw_act.ndim == 4:
                pooled = raw_act.mean(dim=(2, 3))
            else:
                pooled = raw_act
            lc   = layer_cfg[name]
            _print_stats(raw_act.float(), f"{name} raw")
            _print_stats(pooled.float(),  f"{name} pooled")

            N    = pooled.shape[1]
            size = find_cortical_sheet_size(N)
            H, W = size.height, size.width
            H_d  = max(1, round(H / lc["factor_h"]))
            W_d  = max(1, round(W / lc["factor_w"]))
            sheet = pooled[:, :H*W].reshape(B, 1, H, W)
            flat  = F.adaptive_avg_pool2d(sheet, (H_d, W_d)).reshape(B, -1)
            probs = flat.softmax(dim=-1)
            _print_stats(flat.float(),  f"{name} downsampled")
            _print_stats(probs.float(), f"{name} softmax")
            M    = flat.shape[1]
            bm   = probs.mean(0)
            kl   = F.kl_div(torch.full_like(bm, -math.log(M)), bm, reduction="sum")
            ent  = -(probs * (probs + 1e-10).log()).sum(-1).mean()
            print(f"      {name}  factor_h={lc['factor_h']}  factor_w={lc['factor_w']}  "
                  f"lambda_kl={lc['lambda_kl']}  lambda_entropy={lc['lambda_entropy']}  "
                  f"temperature={lc.get('temperature', 3.0)}")
            print(f"      {name}  KL={kl.item():.4g}  entropy={ent.item():.4g}")

        # figure
        n_lay  = len(TOPO_LAYER_NAMES)
        n_cols = 4 * n_lay   # raw | down | probs | hist per layer
        fig, axes = plt.subplots(
            B, n_cols,
            figsize=(n_cols * 1.8, B * 1.8),
            squeeze=False,
        )

        col_titles = []
        for lname in TOPO_LAYER_NAMES:
            col_titles += [f"{lname}\nraw", f"{lname}\ndownsampled",
                           f"{lname}\nsoftmax p", f"{lname}\nhist"]
        for c, t in enumerate(col_titles):
            axes[0, c].set_title(t, fontsize=7, pad=2)

        for row in range(B):
            for l_idx, lname in enumerate(TOPO_LAYER_NAMES):
                act = batch_acts[lname][row]
                act1d = act.mean(dim=(1, 2)) if act.ndim == 3 else act
                raw, down, probs = _sheet_decompose(act1d.float(), lname)

                bc = l_idx * 4

                def _imshow(ax, data, cmap):
                    d = data.numpy() if isinstance(data, torch.Tensor) else data
                    vabs = max(abs(float(d.min())), abs(float(d.max())), 1e-6)
                    if cmap == "RdBu_r":
                        ax.imshow(d, cmap=cmap, vmin=-vabs, vmax=vabs)
                    else:
                        ax.imshow(d, cmap=cmap)
                    ax.axis("off")

                _imshow(axes[row, bc + 0], raw,   "RdBu_r")
                _imshow(axes[row, bc + 1], down,  "RdBu_r")
                _imshow(axes[row, bc + 2], probs, "hot")

                ax_h = axes[row, bc + 3]
                vals = act1d.numpy()
                has_nan = np.isnan(vals).any()
                ax_h.hist(vals[~np.isnan(vals)], bins=30,
                          color="#cc3333" if has_nan else "steelblue",
                          edgecolor="none")
                if has_nan:
                    ax_h.set_title("NaN!", color="red", fontsize=7)
                ax_h.set_xticks([])
                ax_h.set_yticks([])
                ax_h.spines[["top","right"]].set_visible(False)

        # dividers between layer groups
        for l_idx in range(1, n_lay):
            x = (l_idx * 4) / n_cols
            fig.add_artist(plt.Line2D([x, x], [0, 1],
                                      transform=fig.transFigure,
                                      color="black", linewidth=1.0, linestyle="--"))

        layer_info = "  ".join(
            f"{n}(fh={layer_cfg[n]['factor_h']},fw={layer_cfg[n]['factor_w']}"
            f",λkl={layer_cfg[n]['lambda_kl']},λent={layer_cfg[n]['lambda_entropy']})"
            for n in TOPO_LAYER_NAMES
        )
        plt.suptitle(
            f"Cortical sheet debug — model={label}\n"
            f"{layer_info}\n"
            "Cols per layer: raw sheet | downsampled | softmax p | act histogram",
            y=1.003, fontsize=9,
        )
        plt.tight_layout()
        path = out_dir / f"debug_cortical_sheets_{label}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved debug figure -> {path}")


# -- Selectivity visualisation -------------------------------------------------

def _t_values(hook_outputs: torch.Tensor, labels: torch.Tensor,
              target_class: int) -> torch.Tensor:
    """Welch t-statistic per unit: target class vs. all others."""
    if hook_outputs.ndim > 2:
        hook_outputs = hook_outputs.mean(dim=tuple(range(2, hook_outputs.ndim)))
    mask   = labels == target_class
    target = hook_outputs[mask]
    other  = hook_outputs[~mask]
    if target.size(0) < 2 or other.size(0) < 2:
        return torch.zeros(hook_outputs.shape[1])
    n_t, n_o     = target.size(0), other.size(0)
    mu_t, mu_o   = target.mean(0), other.mean(0)
    var_t, var_o = target.var(0, unbiased=True), other.var(0, unbiased=True)
    se = (var_t / n_t + var_o / n_o).clamp(min=1e-12).sqrt()
    return ((mu_t - mu_o) / se).cpu()


def _collect_activations(model: SimpleNN,
                         val_loader: DataLoader,
                         device: str) -> tuple:
    """Run val set through model; return (all_acts dict, all_labels tensor)."""
    acts_store = {name: [] for name in VIS_LAYER_NAMES}
    lbls_store = []
    handles    = []

    for name in VIS_LAYER_NAMES:
        layer = getattr(model, name)
        def _make_hook(n):
            def _h(_mod, _inp, out):
                acts_store[n].append(out.detach().cpu())
            return _h
        handles.append(layer.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        for imgs, lbls in val_loader:
            model(imgs.to(device))
            lbls_store.append(lbls.cpu())

    for h in handles:
        h.remove()

    return ({n: torch.cat(v) for n, v in acts_store.items()},
            torch.cat(lbls_store))


def save_selectivity_maps(model: SimpleNN,
                          val_loader: DataLoader,
                          out_dir: Path,
                          device: str,
                          tag: str = "") -> None:
    """Save a 10-row x 5-col selectivity figure (example + one map per layer)."""
    print(f"  Collecting activations for '{tag}' ...")
    all_acts, all_labels = _collect_activations(model, val_loader, device)

    for name in VIS_LAYER_NAMES:
        a = all_acts[name]
        N = a.shape[1]
        size = find_cortical_sheet_size(N)
        print(f"    {name}: raw {tuple(a.shape)} -> cortical sheet {size.height}x{size.width}")

    n_classes = len(FMNIST_CLASSES)
    n_cols    = 1 + len(VIS_LAYER_NAMES)

    fig, axes = plt.subplots(
        n_classes, n_cols,
        figsize=(n_cols * 2.5, n_classes * 2.5),
        gridspec_kw={"width_ratios": [1] + [2] * len(VIS_LAYER_NAMES)},
    )
    axes[0, 0].set_title("Example", fontsize=9, pad=3)
    for col, name in enumerate(VIS_LAYER_NAMES, start=1):
        marker = "" if name in TOPO_LAYER_NAMES else " (no reg)"
        axes[0, col].set_title(f"{name}{marker}", fontsize=9, pad=3)

    # One example image per class
    example_imgs = {}
    for imgs, lbls in val_loader:
        for cls in range(n_classes):
            if cls not in example_imgs:
                idx = (lbls == cls).nonzero(as_tuple=True)[0]
                if len(idx):
                    example_imgs[cls] = imgs[idx[0]].squeeze().numpy()
        if len(example_imgs) == n_classes:
            break

    for row, cls in enumerate(range(n_classes)):
        ax = axes[row, 0]
        if cls in example_imgs:
            ax.imshow(example_imgs[cls], cmap="gray")
        ax.set_ylabel(FMNIST_CLASSES[cls], fontsize=8, rotation=0,
                      labelpad=50, va="center")
        ax.axis("off")

        for col, name in enumerate(VIS_LAYER_NAMES, start=1):
            a    = all_acts[name]
            N    = a.shape[1]
            size = find_cortical_sheet_size(N)
            tv   = _t_values(a, all_labels, cls)
            tmap = tv[:size.height * size.width].reshape(size.height, size.width).numpy()
            vmax = max(abs(tmap.min()), abs(tmap.max()), 0.1)
            ax   = axes[row, col]
            im   = ax.imshow(tmap, cmap="RdGy_r", vmin=-vmax, vmax=vmax)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.1f")

    suffix = f"_{tag}" if tag else ""
    plt.suptitle(f"Fashion-MNIST Category Selectivity — {tag}", y=1.002, fontsize=12)
    plt.tight_layout()
    path = out_dir / f"selectivity_map{suffix}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved -> {path}")


def _act_to_cortical_sheet(act: torch.Tensor) -> np.ndarray:
    """Convert a single-sample activation tensor to a 2-D cortical sheet (H, W).

    * Conv (C, H, W)  -> global-avg-pool -> (C,) -> cortical sheet
    * Linear (N,)     -> cortical sheet directly
    Values are min-max normalised to [0, 1] for display.
    """
    a = act.detach().cpu().float()
    if a.ndim == 3:                  # (C, H, W) conv output
        a = a.mean(dim=(1, 2))       # (C,)
    elif a.ndim > 1:
        a = a.squeeze()

    a = a.flatten()
    size = find_cortical_sheet_size(a.shape[0])
    a    = a[: size.height * size.width]
    lo, hi = a.min(), a.max()
    if hi > lo:
        a = (a - lo) / (hi - lo)
    sheet = a.reshape(size.height, size.width).numpy()
    return sheet


def save_activation_cortical_sheets(
    models: dict,
    val_loader: DataLoader,
    out_dir: Path,
    device: str,
) -> None:
    """For one example per class, show the cortical sheet of activations at every layer.

    Parameters
    ----------
    models : dict mapping label -> SimpleNN, e.g.
             {"topo": ..., "topo-only": ..., "baseline": ...}

    Figure layout  (10 rows × (1 + 2*n_models) cols):
      Col 0           : example image
      Cols 1-2, 3-4   : per-model activations — fc1 | fc2

    Each cell is min-max normalised and shown with the 'hot' colormap.
    """
    print("  Collecting single-sample activations for cortical-sheet figure ...")

    # --- collect one image per class ------------------------------------------
    example_imgs   = {}   # cls -> (1, 1, 28, 28)
    example_labels = {}
    for imgs, lbls in val_loader:
        for cls in range(len(FMNIST_CLASSES)):
            if cls not in example_imgs:
                idx = (lbls == cls).nonzero(as_tuple=True)[0]
                if len(idx):
                    example_imgs[cls]   = imgs[idx[0]:idx[0]+1]   # (1,1,28,28)
                    example_labels[cls] = cls
        if len(example_imgs) == len(FMNIST_CLASSES):
            break

    # --- helper: get per-layer activations for one image ----------------------
    def _get_layer_acts(model: SimpleNN, img: torch.Tensor) -> dict:
        store   = {}
        handles = []
        for name in VIS_LAYER_NAMES:
            layer = getattr(model, name)
            def _make_hook(n):
                def _h(_m, _i, out):
                    store[n] = out[0]   # drop batch dim -> (C,H,W) or (N,)
                return _h
            handles.append(layer.register_forward_hook(_make_hook(name)))
        model.eval()
        with torch.no_grad():
            model(img.to(device))
        for h in handles:
            h.remove()
        return store

    model_items = list(models.items())   # [(label, model), ...]
    n_classes  = len(FMNIST_CLASSES)
    n_lay      = len(VIS_LAYER_NAMES)
    n_models   = len(model_items)
    n_cols     = 1 + n_models * n_lay

    fig, axes = plt.subplots(
        n_classes, n_cols,
        figsize=(n_cols * 1.8, n_classes * 1.8),
    )

    # Column headers
    axes[0, 0].set_title("Input", fontsize=8, pad=3)
    for m_idx, (lbl, _) in enumerate(model_items):
        for i, name in enumerate(VIS_LAYER_NAMES):
            axes[0, 1 + m_idx * n_lay + i].set_title(
                f"{lbl}\n{name}", fontsize=7, pad=3
            )

    # Dividers between model groups
    for m_idx in range(1, n_models):
        div_x = (1 + m_idx * n_lay) / n_cols
        fig.add_artist(plt.Line2D([div_x, div_x], [0, 1],
                                  transform=fig.transFigure,
                                  color="steelblue", linestyle="--", linewidth=1.2))

    for row, cls in enumerate(range(n_classes)):
        img_tensor = example_imgs[cls]
        model_acts = {lbl: _get_layer_acts(mdl, img_tensor)
                      for lbl, mdl in model_items}

        # Col 0: input image
        ax = axes[row, 0]
        ax.imshow(img_tensor[0, 0].numpy(), cmap="gray")
        ax.set_ylabel(FMNIST_CLASSES[cls], fontsize=7, rotation=0,
                      labelpad=42, va="center")
        ax.axis("off")

        for m_idx, (lbl, _) in enumerate(model_items):
            acts = model_acts[lbl]
            for i, name in enumerate(VIS_LAYER_NAMES):
                sheet = _act_to_cortical_sheet(acts[name])
                axes[row, 1 + m_idx * n_lay + i].imshow(sheet, cmap="hot", vmin=0, vmax=1)
                axes[row, 1 + m_idx * n_lay + i].axis("off")

    model_labels = "  |  ".join(f"{lbl} (cols {1+i*n_lay}-{i*n_lay+n_lay})"
                                for i, (lbl, _) in enumerate(model_items))
    plt.suptitle(
        f"Activation Cortical Sheets — {model_labels}\n"
        "One example per Fashion-MNIST category  |  min-max normalised per cell",
        y=1.003, fontsize=9,
    )
    plt.tight_layout()
    path = out_dir / "activation_cortical_sheets.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved activation cortical sheets -> {path}")


def save_comparison_figure(models: dict,
                           val_loader: DataLoader,
                           out_dir: Path,
                           device: str) -> None:
    """Side-by-side selectivity comparison across all trained models.

    Parameters
    ----------
    models : dict mapping label -> SimpleNN, ordered as desired left-to-right,
             e.g. {"topo": ..., "topo-only": ..., "baseline": ...}

    Layout:
      Rows = 10 classes
      Cols = [example | (fc1 | fc2) per model]
    """
    compare_layers = ["fc1", "fc2"]
    model_items    = list(models.items())

    print("  Collecting activations for comparison figure ...")
    all_acts_dict = {}
    all_labels    = None
    for lbl, mdl in model_items:
        acts, lbls = _collect_activations(mdl, val_loader, device)
        all_acts_dict[lbl] = acts
        if all_labels is None:
            all_labels = lbls

    n_classes  = len(FMNIST_CLASSES)
    n_cl       = len(compare_layers)
    n_models   = len(model_items)
    # cols: example | (fc1 fc2) x n_models
    col_labels = (["Example"]
                  + [f"{lbl}/{n}" for lbl, _ in model_items for n in compare_layers])
    n_cols = len(col_labels)

    fig, axes = plt.subplots(
        n_classes, n_cols,
        figsize=(n_cols * 2.5, n_classes * 2.5),
        gridspec_kw={"width_ratios": [1] + [2] * (n_cols - 1)},
    )
    for col, lbl in enumerate(col_labels):
        axes[0, col].set_title(lbl, fontsize=9, pad=3)

    # Dividers between model groups
    for m_idx in range(1, n_models):
        div_x = (1 + m_idx * n_cl) / n_cols
        fig.add_artist(plt.Line2D([div_x, div_x], [0, 1],
                                  transform=fig.transFigure,
                                  color="gray", linestyle="--", linewidth=1))

    example_imgs = {}
    for imgs, lbls in val_loader:
        for cls in range(n_classes):
            if cls not in example_imgs:
                idx = (lbls == cls).nonzero(as_tuple=True)[0]
                if len(idx):
                    example_imgs[cls] = imgs[idx[0]].squeeze().numpy()
        if len(example_imgs) == n_classes:
            break

    for row, cls in enumerate(range(n_classes)):
        axes[row, 0].imshow(example_imgs.get(cls, np.zeros((28, 28))), cmap="gray")
        axes[row, 0].set_ylabel(FMNIST_CLASSES[cls], fontsize=8, rotation=0,
                                labelpad=50, va="center")
        axes[row, 0].axis("off")

        col = 1
        for lbl, _ in model_items:
            for name in compare_layers:
                a    = all_acts_dict[lbl][name]
                N    = a.shape[1]
                size = find_cortical_sheet_size(N)
                tv   = _t_values(a, all_labels, cls)
                tmap = tv[:size.height * size.width].reshape(size.height, size.width).numpy()
                vmax = max(abs(tmap.min()), abs(tmap.max()), 0.1)
                ax   = axes[row, col]
                im   = ax.imshow(tmap, cmap="RdGy_r", vmin=-vmax, vmax=vmax)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.1f")
                col += 1

    model_str = "  vs  ".join(lbl for lbl, _ in model_items)
    plt.suptitle(f"{model_str} — Category Selectivity (fc1 & fc2)",
                 y=1.002, fontsize=11)
    plt.tight_layout()
    path = out_dir / "selectivity_comparison.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved comparison -> {path}")


# -- Config loading ------------------------------------------------------------

_DEFAULT_CONFIG = {
    "data_dir":         None,          # null → BASE_DIR/data
    "output_dir":       None,          # null → BASE_DIR/outputs/train_topo_sparsity
    "hidden_size":      256,
    "epochs":           10,
    "batch_size":       256,
    "lr":               1e-3,
    "device":           "cuda:0",
    "print_freq":       10,
    "resume_topo":      None,
    "resume_topo_only": None,
    "resume_base":      None,
    "layers": {
        "fc1": {"topo_scale": 10.0, "factor_h": 4.0, "factor_w": 4.0,
                "lambda_kl": 1.0, "lambda_entropy": 1.0, "temperature": 3.0},
    },
}


def get_config():
    """Load config from JSON file, then apply any CLI overrides.

    Usage
    -----
    python train_topo_sparsity.py --config configs/train_topo_sparsity.json

    Any top-level key can be overridden via CLI:
      --epochs 50  --device cuda:1  --data-dir /path/to/data  etc.

    Per-layer settings (topo_scale, factor_h, factor_w, lambda_kl,
    lambda_entropy) must be edited in the JSON file.
    """
    p = argparse.ArgumentParser(
        description="Train SimpleNN with topographic regularisation on Fashion-MNIST.\n"
                    "All hyperparameters live in a JSON config file; top-level\n"
                    "keys can be overridden via CLI arguments."
    )
    default_cfg = str(BASE_DIR / "configs" / "train_topo_sparsity.json")
    p.add_argument("--config",           type=str, default=default_cfg,
                   help="Path to JSON config file")
    # top-level CLI overrides (all optional — None means 'use JSON value')
    p.add_argument("--data-dir",         type=str,   default=None)
    p.add_argument("--output-dir",       type=str,   default=None)
    p.add_argument("--hidden-size",      type=int,   default=None)
    p.add_argument("--epochs",           type=int,   default=None)
    p.add_argument("--batch-size",       type=int,   default=None)
    p.add_argument("--lr",               type=float, default=None)
    p.add_argument("--device",           type=str,   default=None)
    p.add_argument("--print-freq",       type=int,   default=None)
    p.add_argument("--resume-topo",      type=str,   default=None)
    p.add_argument("--resume-topo-only", type=str,   default=None)
    p.add_argument("--resume-base",      type=str,   default=None)
    cli = p.parse_args()

    # Start from built-in defaults, overlay JSON file, then overlay CLI
    import copy as _copy
    cfg = _copy.deepcopy(_DEFAULT_CONFIG)

    cfg_path = Path(cli.config)
    if cfg_path.exists():
        with open(cfg_path) as fh:
            file_cfg = json.load(fh)
        # deep-merge layers sub-dict
        for key, val in file_cfg.items():
            if key == "layers" and isinstance(val, dict):
                for lname, lvals in val.items():
                    if lname in cfg["layers"]:
                        cfg["layers"][lname].update(lvals)
                    else:
                        cfg["layers"][lname] = lvals
            else:
                cfg[key] = val
        print(f"Config loaded from: {cfg_path}")
    else:
        print(f"Config file not found ({cfg_path}), using built-in defaults.")

    # CLI overrides (argparse uses underscores, JSON uses underscores too)
    cli_map = {
        "data_dir":         cli.data_dir,
        "output_dir":       cli.output_dir,
        "hidden_size":      cli.hidden_size,
        "epochs":           cli.epochs,
        "batch_size":       cli.batch_size,
        "lr":               cli.lr,
        "device":           cli.device,
        "print_freq":       cli.print_freq,
        "resume_topo":      cli.resume_topo,
        "resume_topo_only": cli.resume_topo_only,
        "resume_base":      cli.resume_base,
    }
    for key, val in cli_map.items():
        if val is not None:
            cfg[key] = val

    # Fill in path defaults
    if cfg["data_dir"]   is None:
        cfg["data_dir"]   = str(BASE_DIR / "data")
    if cfg["output_dir"] is None:
        cfg["output_dir"] = str(OUTPUT_DIR)

    return cfg


# -- Single-model training loop ------------------------------------------------

def run_training(
    label: str,
    model: SimpleNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int,
    ckpt_dir: Path,
    device: str,
    print_freq: int,
    topo_loss=None,          # None => baseline (CE only)
    layer_cfg: dict = None,  # {layer_name: {factor_h, factor_w, lambda_kl, lambda_entropy}}
    start_epoch: int = 0,
    best_acc: float = 0.0,
) -> SimpleNN:
    """Train one model; return the model with best-checkpoint weights loaded."""

    act_store: dict = {name: None for name in TOPO_LAYER_NAMES}
    hook_handles    = []
    if topo_loss is not None:
        for name in TOPO_LAYER_NAMES:
            layer = getattr(model, name)
            def _make_hook(n):
                def _h(_mod, _inp, out):
                    act_store[n] = out
                return _h
            hook_handles.append(layer.register_forward_hook(_make_hook(name)))

    for epoch in range(start_epoch, epochs):
        model.train()
        sum_ce = sum_topo = 0.0
        sum_kl  = {n: 0.0 for n in TOPO_LAYER_NAMES}
        sum_ent = {n: 0.0 for n in TOPO_LAYER_NAMES}
        n_correct = n_total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            ce     = criterion(logits, labels)

            extra = torch.tensor(0.0, device=device)
            if topo_loss is not None:
                topo      = topo_loss.compute(model=model, reduce_mean=True)
                sum_topo += topo.item() * imgs.size(0)
                extra     = extra + topo

                for name in TOPO_LAYER_NAMES:
                    act = act_store[name]
                    if act is not None:
                        lc  = layer_cfg[name]
                        kl, ent = cortical_sparsity_losses(
                            act, lc["factor_h"], lc["factor_w"],
                            lc.get("temperature", 3.0)
                        )
                        extra = (extra
                                 + lc["lambda_kl"]      * kl
                                 + lc["lambda_entropy"] * ent)
                        sum_kl[name]  += kl.item()  * imgs.size(0)
                        sum_ent[name] += ent.item() * imgs.size(0)

            loss = ce + extra
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            sum_ce    += ce.item() * bs
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_total   += bs

        # Validation
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_correct += (model(imgs).argmax(1) == labels).sum().item()
                val_total   += labels.size(0)
        model.train()

        val_acc   = 100 * val_correct / val_total
        train_acc = 100 * n_correct   / n_total

        if (epoch + 1) % print_freq == 0 or epoch == epochs - 1:
            n = n_total
            if topo_loss is not None:
                kl_str  = "  ".join(f"{nm}:{sum_kl[nm]/n:.3f}"  for nm in TOPO_LAYER_NAMES)
                ent_str = "  ".join(f"{nm}:{sum_ent[nm]/n:.3f}" for nm in TOPO_LAYER_NAMES)
                print(
                    f"[{label}] Epoch [{epoch+1:3d}/{epochs}]  "
                    f"CE={sum_ce/n:.4f}  "
                    f"\033[93mTopo={sum_topo/n:.6f}\033[0m  "
                    f"Train={train_acc:.1f}%  \033[92mVal={val_acc:.1f}%\033[0m\n"
                    f"  KL  [{kl_str}]\n"
                    f"  Ent [{ent_str}]"
                )
            else:
                print(
                    f"[{label}] Epoch [{epoch+1:3d}/{epochs}]  "
                    f"CE={sum_ce/n:.4f}  "
                    f"Train={train_acc:.1f}%  \033[92mVal={val_acc:.1f}%\033[0m"
                )

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc":  best_acc,
            }, ckpt_dir / f"best_{label}.pt")

    torch.save({
        "epoch":     epochs - 1,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_acc":  best_acc,
    }, ckpt_dir / f"last_{label}.pt")

    for h in hook_handles:
        h.remove()

    print(f"\n[{label}] Training complete.  Best val acc: {best_acc:.1f}%")

    # Load best weights before returning
    best_ckpt = torch.load(ckpt_dir / f"best_{label}.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    return model


# -- Main entry ----------------------------------------------------------------

def _build_topo_loss(model: SimpleNN, layer_cfg: dict) -> TopoLoss:
    """Construct a TopoLoss for fc1 only."""
    return TopoLoss(losses=[
        LaplacianPyramid.from_layer(
            model=model, layer=model.fc1,
            factor_h=layer_cfg["fc1"]["factor_h"],
            factor_w=layer_cfg["fc1"]["factor_w"],
            scale=layer_cfg["fc1"]["topo_scale"],
        )
    ])


def train(cfg: dict):
    # Print resolved config
    print("=" * 65)
    print("  EFFECTIVE CONFIG")
    print("=" * 65)
    print(json.dumps(
        {k: v for k, v in cfg.items() if k != "layers"},
        indent=2,
    ))
    print("  layers:")
    for lname, lvals in cfg["layers"].items():
        print(f"    {lname}: {json.dumps(lvals)}")
    print()

    device   = cfg["device"] if torch.cuda.is_available() else "cpu"
    out_dir  = Path(cfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_ds = datasets.FashionMNIST(cfg["data_dir"], train=True,  download=True, transform=transform)
    val_ds   = datasets.FashionMNIST(cfg["data_dir"], train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f"Dataset: {len(train_ds):,} train | {len(val_ds):,} val | 10 classes\n")

    layer_cfg = cfg["layers"]  # shorthand

    # ── Topo model (TopoLoss + KL + entropy sparsity) ───────────────────────
    print("=" * 65)
    print("  TOPO MODEL  (TopoLoss + sparsity on fc1)")
    print("=" * 65)
    topo_model = SimpleNN(hidden_size=cfg.get("hidden_size", 256)).to(device)
    topo_optim = optim.Adam(topo_model.parameters(), lr=cfg["lr"])
    topo_start, topo_best = 0, 0.0

    if cfg.get("resume_topo"):
        ckpt = torch.load(cfg["resume_topo"], map_location=device)
        topo_model.load_state_dict(ckpt["model"])
        topo_optim.load_state_dict(ckpt["optimizer"])
        topo_start = ckpt["epoch"] + 1
        topo_best  = ckpt.get("best_acc", 0.0)
        print(f"  Resumed from epoch {topo_start}")

    criterion  = nn.CrossEntropyLoss().to(device)
    topo_model = run_training(
        label="topo",
        model=topo_model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=topo_optim,
        epochs=cfg["epochs"], ckpt_dir=ckpt_dir, device=device,
        print_freq=cfg["print_freq"],
        topo_loss=_build_topo_loss(topo_model, layer_cfg),
        layer_cfg=layer_cfg,
        start_epoch=topo_start, best_acc=topo_best,
    )

    # ── Topo-only model (TopoLoss, no sparsity) ──────────────────────────────
    print("\n" + "=" * 65)
    print("  TOPO-ONLY MODEL  (TopoLoss only on fc1, no KL/entropy sparsity)")
    print("=" * 65)
    topo_only_model = SimpleNN(hidden_size=cfg.get("hidden_size", 256)).to(device)
    topo_only_optim = optim.Adam(topo_only_model.parameters(), lr=cfg["lr"])
    topo_only_start, topo_only_best = 0, 0.0

    if cfg.get("resume_topo_only"):
        ckpt = torch.load(cfg["resume_topo_only"], map_location=device)
        topo_only_model.load_state_dict(ckpt["model"])
        topo_only_optim.load_state_dict(ckpt["optimizer"])
        topo_only_start = ckpt["epoch"] + 1
        topo_only_best  = ckpt.get("best_acc", 0.0)
        print(f"  Resumed from epoch {topo_only_start}")

    # zero-out lambda_kl / lambda_entropy for the topo-only variant
    import copy as _copy
    topo_only_layer_cfg = {
        name: {**vals, "lambda_kl": 0.0, "lambda_entropy": 0.0}
        for name, vals in layer_cfg.items()
    }
    topo_only_model = run_training(
        label="topo_only",
        model=topo_only_model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=topo_only_optim,
        epochs=cfg["epochs"], ckpt_dir=ckpt_dir, device=device,
        print_freq=cfg["print_freq"],
        topo_loss=_build_topo_loss(topo_only_model, layer_cfg),
        layer_cfg=topo_only_layer_cfg,    # lambdas zeroed
        start_epoch=topo_only_start, best_acc=topo_only_best,
    )

    # ── Baseline model (CE only) ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  BASELINE MODEL  (CE only, no TopoLoss, no sparsity)")
    print("=" * 65)
    base_model = SimpleNN(hidden_size=cfg.get("hidden_size", 256)).to(device)
    base_optim = optim.Adam(base_model.parameters(), lr=cfg["lr"])
    base_start, base_best = 0, 0.0

    if cfg.get("resume_base"):
        ckpt = torch.load(cfg["resume_base"], map_location=device)
        base_model.load_state_dict(ckpt["model"])
        base_optim.load_state_dict(ckpt["optimizer"])
        base_start = ckpt["epoch"] + 1
        base_best  = ckpt.get("best_acc", 0.0)
        print(f"  Resumed from epoch {base_start}")

    base_model = run_training(
        label="baseline",
        model=base_model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=base_optim,
        epochs=cfg["epochs"], ckpt_dir=ckpt_dir, device=device,
        print_freq=cfg["print_freq"],
        topo_loss=None,    # CE only — layer_cfg unused when topo_loss is None
        start_epoch=base_start, best_acc=base_best,
    )

    # ── Visualisations ───────────────────────────────────────────────────────
    all_models = {
        "topo":      topo_model,
        "topo-only": topo_only_model,
        "baseline":  base_model,
    }
    print("\n" + "=" * 65)
    print("  Generating selectivity maps ...")
    print("=" * 65)
    for lbl, mdl in all_models.items():
        save_selectivity_maps(mdl, val_loader, out_dir, device, tag=lbl)
    save_comparison_figure(all_models, val_loader, out_dir, device)
    save_activation_cortical_sheets(all_models, val_loader, out_dir, device)
    save_debug_cortical_sheets(
        all_models, val_loader,
        layer_cfg=layer_cfg,
        out_dir=out_dir, device=device,
    )

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    cfg = get_config()
    train(cfg)
