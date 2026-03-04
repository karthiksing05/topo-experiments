"""
Topographic regularisation on ResNet-18 trained on ImageNet.

Three conditions trained and compared:
  topo      — TopoLoss + KL/entropy sparsity on layer1…layer4
  topo-only — TopoLoss only (no sparsity) on layer1…layer4
  baseline  — Cross-entropy only

TopoLoss is applied to the four residual-block groups (layer1–4).
The final classifier (fc) is captured for visualisation but not regularised.

All hyperparameters live in a JSON config file (configs/train_topo_sparsity_large.json).
Top-level keys can be overridden via CLI; per-layer keys must be edited in the JSON.
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
import torchvision.models as models
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset

from topoloss import LaplacianPyramid, TopoLoss
from topoloss.core import find_cortical_sheet_size

# -- Defaults ------------------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs" / "train_topo_sparsity_large"

# Residual groups — captured for visualisation + regularisation
VIS_LAYER_NAMES  = ["layer1", "layer2", "layer3", "layer4", "fc"]
# Layers that get TopoLoss + KL/entropy  (fc excluded from regularisation)
TOPO_LAYER_NAMES = ["layer1", "layer2", "layer3", "layer4"]

# ImageNet normalisation constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# -- Cortical-sheet activation regularisers ------------------------------------

def cortical_sparsity_losses(activations: torch.Tensor,
                             factor_h: float, factor_w: float,
                             temperature: float = 3.0) -> tuple:
    """KL-from-uniform + per-sample entropy on the downsampled cortical sheet.

    Spatial activations (B,C,H,W) are globally avg-pooled to (B,C) first so
    that output channels form the topographic axis.
    """

    if activations.ndim == 4:
        activations = activations.mean(dim=(2, 3))   # (B, C)

    B, N  = activations.shape
    size  = find_cortical_sheet_size(N)
    H, W  = size.height, size.width

    sheet  = activations.reshape(B, 1, H, W)
    H_down = max(1, round(H / factor_h))
    W_down = max(1, round(W / factor_w))
    flat   = F.adaptive_avg_pool2d(sheet, (H_down, W_down)).reshape(B, -1)
    M      = flat.shape[1]

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
    """For each model save a debug figure of per-layer cortical sheets.

    One PNG per model: debug_cortical_sheets_{label}.png
    Rows = up to 8 val samples; column groups per TOPO layer:
      raw sheet | downsampled | softmax probs | activation histogram
    Prints NaN/inf stats and KL/entropy values to stdout.
    """
    print("\n" + "=" * 65)
    print("  CORTICAL SHEET DEBUG")
    print("=" * 65)

    n_show = 8

    def _collect_one_batch(model):
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

        for name in TOPO_LAYER_NAMES:
            raw_act = batch_acts[name][:B]
            pooled  = raw_act.mean(dim=(2, 3)) if raw_act.ndim == 4 else raw_act
            lc      = layer_cfg[name]
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
            M   = flat.shape[1]
            bm  = probs.mean(0)
            kl  = F.kl_div(torch.full_like(bm, -math.log(M)), bm, reduction="sum")
            ent = -(probs * (probs + 1e-10).log()).sum(-1).mean()
            print(f"      {name}  factor_h={lc['factor_h']}  factor_w={lc['factor_w']}  "
                  f"lambda_kl={lc['lambda_kl']}  lambda_entropy={lc['lambda_entropy']}  "
                  f"temperature={lc.get('temperature', 3.0)}")
            print(f"      {name}  KL={kl.item():.4g}  entropy={ent.item():.4g}")

        n_lay  = len(TOPO_LAYER_NAMES)
        n_cols = 4 * n_lay
        fig, axes = plt.subplots(B, n_cols, figsize=(n_cols * 1.8, B * 1.8), squeeze=False)

        col_titles = []
        for lname in TOPO_LAYER_NAMES:
            col_titles += [f"{lname}\nraw", f"{lname}\ndown",
                           f"{lname}\nprobs", f"{lname}\nhist"]
        for c, t in enumerate(col_titles):
            axes[0, c].set_title(t, fontsize=6, pad=2)

        for row in range(B):
            for l_idx, lname in enumerate(TOPO_LAYER_NAMES):
                act   = batch_acts[lname][row]
                act1d = act.mean(dim=(1, 2)) if act.ndim == 3 else act
                raw, down, probs = _sheet_decompose(act1d.float(), lname)
                bc = l_idx * 4

                def _imshow(ax, data, cmap):
                    d    = data.numpy() if isinstance(data, torch.Tensor) else data
                    vabs = max(abs(float(d.min())), abs(float(d.max())), 1e-6)
                    ax.imshow(d, cmap=cmap, vmin=-vabs if cmap == "RdBu_r" else d.min(),
                              vmax=vabs if cmap == "RdBu_r" else d.max())
                    ax.axis("off")

                _imshow(axes[row, bc + 0], raw,   "RdBu_r")
                _imshow(axes[row, bc + 1], down,  "RdBu_r")
                _imshow(axes[row, bc + 2], probs, "hot")

                ax_h  = axes[row, bc + 3]
                vals  = act1d.numpy()
                valid = vals[~np.isnan(vals)]
                has_nan = len(valid) < len(vals)
                ax_h.hist(valid, bins=30,
                          color="#cc3333" if has_nan else "steelblue",
                          edgecolor="none")
                if has_nan:
                    ax_h.set_title("NaN!", color="red", fontsize=6)
                ax_h.set_xticks([])
                ax_h.set_yticks([])
                ax_h.spines[["top", "right"]].set_visible(False)

        for l_idx in range(1, n_lay):
            x = (l_idx * 4) / n_cols
            fig.add_artist(plt.Line2D([x, x], [0, 1], transform=fig.transFigure,
                                      color="black", linewidth=1.0, linestyle="--"))

        layer_info = "  ".join(
            f"{n}(fh={layer_cfg[n]['factor_h']},fw={layer_cfg[n]['factor_w']}"
            f",λkl={layer_cfg[n]['lambda_kl']},λent={layer_cfg[n]['lambda_entropy']}"
            f",T={layer_cfg[n].get('temperature', 3.0)})"
            for n in TOPO_LAYER_NAMES
        )
        plt.suptitle(
            f"Cortical sheet debug — model={label}\n{layer_info}\n"
            "Cols per layer: raw | downsampled | softmax p | activation hist",
            y=1.003, fontsize=8,
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


def _collect_activations(model, val_loader, device):
    """Run val loader through model; return (acts_dict, all_labels)."""
    acts_store = {name: [] for name in VIS_LAYER_NAMES}
    lbls_store = []
    handles    = []

    for name in VIS_LAYER_NAMES:
        layer = getattr(model, name)
        def _make_hook(n):
            def _h(_mod, _inp, out):
                # pool spatial dims for conv layers so shape is always (N,)
                a = out.detach().cpu()
                if a.ndim == 4:
                    a = a.mean(dim=(2, 3))
                acts_store[n].append(a)
            return _h
        handles.append(layer.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        for imgs, lbls in val_loader:
            model(imgs.to(device))
            lbls_store.append(lbls)

    for h in handles:
        h.remove()

    return ({n: torch.cat(v) for n, v in acts_store.items()},
            torch.cat(lbls_store))


def save_selectivity_maps(model, val_loader, out_dir, device,
                          vis_class_indices, class_names, tag=""):
    """Save selectivity t-maps for a chosen subset of ImageNet classes.

    vis_class_indices : list of class indices (0-999) to include as rows.
    class_names       : list of 1000 ImageNet class name strings.
    """
    print(f"  Collecting activations for '{tag}' ...")
    all_acts, all_labels = _collect_activations(model, val_loader, device)

    for name in VIS_LAYER_NAMES:
        a    = all_acts[name]
        N    = a.shape[1]
        size = find_cortical_sheet_size(N)
        print(f"    {name}: raw {tuple(a.shape)} -> cortical {size.height}×{size.width}")

    n_vis  = len(vis_class_indices)
    n_cols = 1 + len(VIS_LAYER_NAMES)

    fig, axes = plt.subplots(
        n_vis, n_cols,
        figsize=(n_cols * 2.5, n_vis * 2.5),
        gridspec_kw={"width_ratios": [1] + [2] * len(VIS_LAYER_NAMES)},
    )
    if n_vis == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title("Example", fontsize=9, pad=3)
    for col, name in enumerate(VIS_LAYER_NAMES, start=1):
        marker = "" if name in TOPO_LAYER_NAMES else " (no reg)"
        axes[0, col].set_title(f"{name}{marker}", fontsize=9, pad=3)

    # Collect one example image per vis class
    example_imgs = {}
    for imgs, lbls in val_loader:
        for cls in vis_class_indices:
            if cls not in example_imgs:
                idx = (lbls == cls).nonzero(as_tuple=True)[0]
                if len(idx):
                    example_imgs[cls] = imgs[idx[0]].numpy()
        if len(example_imgs) == n_vis:
            break

    for row, cls in enumerate(vis_class_indices):
        ax = axes[row, 0]
        if cls in example_imgs:
            img = example_imgs[cls]
            # ImageNet images: (3, 224, 224) -> (224, 224, 3) for imshow
            img = np.transpose(img, (1, 2, 0))
            img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img = np.clip(img, 0, 1)
            ax.imshow(img)
        lbl_str = class_names[cls] if class_names else str(cls)
        ax.set_ylabel(lbl_str[:18], fontsize=7, rotation=0,
                      labelpad=60, va="center")
        ax.axis("off")

        for col, name in enumerate(VIS_LAYER_NAMES, start=1):
            a    = all_acts[name]
            N    = a.shape[1]
            size = find_cortical_sheet_size(N)
            tv   = _t_values(a, all_labels, cls)
            tmap = tv[:size.height * size.width].reshape(size.height, size.width).numpy()
            vmax = max(abs(tmap.min()), abs(tmap.max()), 0.1)
            axes[row, col].imshow(tmap, cmap="RdGy_r", vmin=-vmax, vmax=vmax)
            axes[row, col].axis("off")

    suffix = f"_{tag}" if tag else ""
    plt.suptitle(f"ImageNet Category Selectivity — {tag}", y=1.002, fontsize=12)
    plt.tight_layout()
    path = out_dir / f"selectivity_map{suffix}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved -> {path}")


def _act_to_cortical_sheet(act: torch.Tensor) -> np.ndarray:
    """Single-sample activation → 2-D min-max normalised cortical sheet."""
    a = act.detach().cpu().float()
    if a.ndim == 3:               # (C, H, W) spatial output
        a = a.mean(dim=(1, 2))    # (C,)
    elif a.ndim > 1:
        a = a.squeeze()
    a    = a.flatten()
    size = find_cortical_sheet_size(a.shape[0])
    a    = a[: size.height * size.width]
    lo, hi = a.min(), a.max()
    if hi > lo:
        a = (a - lo) / (hi - lo)
    return a.reshape(size.height, size.width).numpy()


def save_activation_cortical_sheets(models, val_loader, out_dir, device,
                                     vis_class_indices, class_names):
    """For one example per vis class show activation cortical sheets across models."""
    print("  Collecting single-sample activations for cortical-sheet figure ...")

    example_imgs = {}
    for imgs, lbls in val_loader:
        for cls in vis_class_indices:
            if cls not in example_imgs:
                idx = (lbls == cls).nonzero(as_tuple=True)[0]
                if len(idx):
                    example_imgs[cls] = imgs[idx[0]:idx[0]+1]
        if len(example_imgs) == len(vis_class_indices):
            break

    def _get_layer_acts(model, img):
        store   = {}
        handles = []
        for name in VIS_LAYER_NAMES:
            layer = getattr(model, name)
            def _make_hook(n):
                def _h(_m, _i, out):
                    store[n] = out[0]
                return _h
            handles.append(layer.register_forward_hook(_make_hook(name)))
        model.eval()
        with torch.no_grad():
            model(img.to(device))
        for h in handles:
            h.remove()
        return store

    model_items = list(models.items())
    n_rows  = len(vis_class_indices)
    n_lay   = len(VIS_LAYER_NAMES)
    n_models = len(model_items)
    n_cols  = 1 + n_models * n_lay

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.6, n_rows * 1.6))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title("Input", fontsize=7, pad=2)
    for m_idx, (lbl, _) in enumerate(model_items):
        for i, name in enumerate(VIS_LAYER_NAMES):
            axes[0, 1 + m_idx * n_lay + i].set_title(
                f"{lbl}\n{name}", fontsize=6, pad=2)

    for m_idx in range(1, n_models):
        div_x = (1 + m_idx * n_lay) / n_cols
        fig.add_artist(plt.Line2D([div_x, div_x], [0, 1],
                                  transform=fig.transFigure,
                                  color="steelblue", linestyle="--", linewidth=1.2))

    for row, cls in enumerate(vis_class_indices):
        img_tensor = example_imgs[cls]
        model_acts = {lbl: _get_layer_acts(mdl, img_tensor)
                      for lbl, mdl in model_items}

        ax = axes[row, 0]
        img = img_tensor[0].numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1)
        ax.imshow(img)
        lbl_str = (class_names[cls] if class_names else str(cls))[:14]
        ax.set_ylabel(lbl_str, fontsize=6, rotation=0, labelpad=46, va="center")
        ax.axis("off")

        for m_idx, (lbl, _) in enumerate(model_items):
            acts = model_acts[lbl]
            for i, name in enumerate(VIS_LAYER_NAMES):
                sheet = _act_to_cortical_sheet(acts[name])
                axes[row, 1 + m_idx * n_lay + i].imshow(sheet, cmap="hot",
                                                          vmin=0, vmax=1)
                axes[row, 1 + m_idx * n_lay + i].axis("off")

    model_labels = "  |  ".join(f"{lbl} (cols {1+i*n_lay}–{i*n_lay+n_lay})"
                                for i, (lbl, _) in enumerate(model_items))
    plt.suptitle(
        f"Activation Cortical Sheets — {model_labels}\n"
        "One example per selected ImageNet class  |  min-max normalised",
        y=1.003, fontsize=8,
    )
    plt.tight_layout()
    path = out_dir / "activation_cortical_sheets.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved activation cortical sheets -> {path}")


def save_comparison_figure(models, val_loader, out_dir, device,
                            vis_class_indices, class_names):
    """Side-by-side t-map comparison (layer1 & layer4) across all models."""
    compare_layers = ["layer1", "layer4"]
    model_items    = list(models.items())

    print("  Collecting activations for comparison figure ...")
    all_acts_dict = {}
    all_labels    = None
    for lbl, mdl in model_items:
        acts, lbls = _collect_activations(mdl, val_loader, device)
        all_acts_dict[lbl] = acts
        if all_labels is None:
            all_labels = lbls

    n_vis  = len(vis_class_indices)
    n_cl   = len(compare_layers)
    n_models = len(model_items)
    col_labels = (["Example"]
                  + [f"{lbl}/{n}" for lbl, _ in model_items for n in compare_layers])
    n_cols = len(col_labels)

    fig, axes = plt.subplots(
        n_vis, n_cols,
        figsize=(n_cols * 2.5, n_vis * 2.5),
        gridspec_kw={"width_ratios": [1] + [2] * (n_cols - 1)},
    )
    if n_vis == 1:
        axes = axes[np.newaxis, :]

    for col, lbl in enumerate(col_labels):
        axes[0, col].set_title(lbl, fontsize=9, pad=3)

    for m_idx in range(1, n_models):
        div_x = (1 + m_idx * n_cl) / n_cols
        fig.add_artist(plt.Line2D([div_x, div_x], [0, 1],
                                  transform=fig.transFigure,
                                  color="gray", linestyle="--", linewidth=1))

    example_imgs = {}
    for imgs, lbls in val_loader:
        for cls in vis_class_indices:
            if cls not in example_imgs:
                idx = (lbls == cls).nonzero(as_tuple=True)[0]
                if len(idx):
                    example_imgs[cls] = imgs[idx[0]].numpy()
        if len(example_imgs) == n_vis:
            break

    for row, cls in enumerate(vis_class_indices):
        ax = axes[row, 0]
        if cls in example_imgs:
            img = np.transpose(example_imgs[cls], (1, 2, 0))
            img = np.clip(img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1)
            ax.imshow(img)
        lbl_str = (class_names[cls] if class_names else str(cls))[:18]
        ax.set_ylabel(lbl_str, fontsize=7, rotation=0, labelpad=60, va="center")
        ax.axis("off")

        col = 1
        for lbl_m, _ in model_items:
            for name in compare_layers:
                a    = all_acts_dict[lbl_m][name]
                N    = a.shape[1]
                size = find_cortical_sheet_size(N)
                tv   = _t_values(a, all_labels, cls)
                tmap = tv[:size.height * size.width].reshape(size.height, size.width).numpy()
                vmax = max(abs(tmap.min()), abs(tmap.max()), 0.1)
                axes[row, col].imshow(tmap, cmap="RdGy_r", vmin=-vmax, vmax=vmax)
                axes[row, col].axis("off")
                col += 1

    model_str = "  vs  ".join(lbl for lbl, _ in model_items)
    plt.suptitle(f"{model_str} — Category Selectivity (layer1 & layer4)",
                 y=1.002, fontsize=11)
    plt.tight_layout()
    path = out_dir / "selectivity_comparison.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved comparison -> {path}")


# -- Config loading ------------------------------------------------------------

_DEFAULT_CONFIG = {
    "data_dir":          None,
    "output_dir":        None,
    "epochs":            90,
    "batch_size":        256,
    "lr":                0.1,
    "lr_schedule":       "cosine",   # "cosine" | "step"
    "momentum":          0.9,
    "weight_decay":      1e-4,
    "amp":               True,
    "pretrained":        False,
    "device":            "cuda:0",
    "print_freq":        50,
    "num_workers":       8,
    "resume_topo":       None,
    "resume_topo_only":  None,
    "resume_base":       None,
    "hf_repo":           "ILSVRC/imagenet-1k",
    "hf_train_size":     100000,
    "hf_val_size":       10000,
    "vis_class_indices": [0, 1, 2, 10, 20, 30, 40, 50,
                          100, 151, 207, 281, 340, 386,
                          440, 530, 609, 701, 817, 950],
    "layers": {
        "layer1": {"topo_scale": 1.0, "factor_h": 4.0, "factor_w": 4.0,
                   "lambda_kl": 0.1, "lambda_entropy": 0.1, "temperature": 3.0},
        "layer2": {"topo_scale": 1.0, "factor_h": 4.0, "factor_w": 4.0,
                   "lambda_kl": 0.1, "lambda_entropy": 0.1, "temperature": 3.0},
        "layer3": {"topo_scale": 1.0, "factor_h": 4.0, "factor_w": 4.0,
                   "lambda_kl": 0.1, "lambda_entropy": 0.1, "temperature": 3.0},
        "layer4": {"topo_scale": 1.0, "factor_h": 4.0, "factor_w": 4.0,
                   "lambda_kl": 0.1, "lambda_entropy": 0.1, "temperature": 3.0},
    },
}


def get_config():
    """Load JSON config, then apply any CLI overrides.

    Per-layer settings must be edited in the JSON file.
    Top-level keys (epochs, lr, device, …) can be overridden via CLI.
    """
    p = argparse.ArgumentParser(
        description="Train ResNet-18 with topographic regularisation on ImageNet."
    )
    default_cfg = str(BASE_DIR / "configs" / "train_topo_sparsity_large.json")
    p.add_argument("--config",           type=str,   default=default_cfg)
    p.add_argument("--data-dir",         type=str,   default=None)
    p.add_argument("--output-dir",       type=str,   default=None)
    p.add_argument("--epochs",           type=int,   default=None)
    p.add_argument("--batch-size",       type=int,   default=None)
    p.add_argument("--lr",               type=float, default=None)
    p.add_argument("--lr-schedule",      type=str,   default=None,
                   choices=["cosine", "step"])
    p.add_argument("--weight-decay",     type=float, default=None)
    p.add_argument("--momentum",         type=float, default=None)
    p.add_argument("--amp",              action="store_true", default=None)
    p.add_argument("--no-amp",           action="store_false", dest="amp")
    p.add_argument("--pretrained",       action="store_true", default=None)
    p.add_argument("--device",           type=str,   default=None)
    p.add_argument("--print-freq",       type=int,   default=None)
    p.add_argument("--num-workers",      type=int,   default=None)
    p.add_argument("--resume-topo",      type=str,   default=None)
    p.add_argument("--resume-topo-only", type=str,   default=None)
    p.add_argument("--resume-base",      type=str,   default=None)
    cli = p.parse_args()

    import copy as _copy
    cfg = _copy.deepcopy(_DEFAULT_CONFIG)

    cfg_path = Path(cli.config)
    if cfg_path.exists():
        with open(cfg_path) as fh:
            file_cfg = json.load(fh)
        for key, val in file_cfg.items():
            if key == "layers" and isinstance(val, dict):
                for lname, lvals in val.items():
                    if lname in cfg["layers"]:
                        cfg["layers"][lname].update(lvals)
                    else:
                        cfg["layers"][lname] = lvals
            elif not key.startswith("_comment"):
                cfg[key] = val
        print(f"Config loaded from: {cfg_path}")
    else:
        print(f"Config file not found ({cfg_path}), using built-in defaults.")

    cli_map = {
        "data_dir":         cli.data_dir,
        "output_dir":       cli.output_dir,
        "epochs":           cli.epochs,
        "batch_size":       cli.batch_size,
        "lr":               cli.lr,
        "lr_schedule":      cli.lr_schedule,
        "weight_decay":     cli.weight_decay,
        "momentum":         cli.momentum,
        "amp":              cli.amp,
        "pretrained":       cli.pretrained,
        "device":           cli.device,
        "print_freq":       cli.print_freq,
        "num_workers":      cli.num_workers,
        "resume_topo":      cli.resume_topo,
        "resume_topo_only": cli.resume_topo_only,
        "resume_base":      cli.resume_base,
    }
    for key, val in cli_map.items():
        if val is not None:
            cfg[key] = val

    if cfg["data_dir"]   is None:
        cfg["data_dir"]   = str(BASE_DIR / "data" / "imagenet")
    if cfg["output_dir"] is None:
        cfg["output_dir"] = str(OUTPUT_DIR)

    return cfg


# -- Single-model training loop ------------------------------------------------

def run_training(
    label: str,
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epochs: int,
    ckpt_dir: Path,
    device: str,
    print_freq: int,
    use_amp: bool,
    topo_loss=None,
    layer_cfg: dict = None,
    start_epoch: int = 0,
    best_acc: float = 0.0,
):
    """Train one model; return model with best-checkpoint weights reloaded."""

    act_store    = {name: None for name in TOPO_LAYER_NAMES}
    hook_handles = []
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

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            with autocast(enabled=use_amp):
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
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            bs = imgs.size(0)
            sum_ce    += ce.item() * bs
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_total   += bs

            if (batch_idx + 1) % print_freq == 0:
                train_acc = 100 * n_correct / n_total
                if topo_loss is not None:
                    print(
                        f"[{label}] E{epoch+1} B{batch_idx+1}  "
                        f"CE={sum_ce/n_total:.4f}  "
                        f"\033[93mTopo={sum_topo/n_total:.4g}\033[0m  "
                        f"Acc={train_acc:.1f}%"
                    )
                else:
                    print(
                        f"[{label}] E{epoch+1} B{batch_idx+1}  "
                        f"CE={sum_ce/n_total:.4f}  Acc={train_acc:.1f}%"
                    )

        scheduler.step()

        # Validation
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_correct += (model(imgs).argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc   = 100 * val_correct  / val_total
        train_acc = 100 * n_correct    / n_total
        n         = n_total

        if topo_loss is not None:
            kl_str  = "  ".join(f"{nm}:{sum_kl[nm]/n:.3f}"  for nm in TOPO_LAYER_NAMES)
            ent_str = "  ".join(f"{nm}:{sum_ent[nm]/n:.3f}" for nm in TOPO_LAYER_NAMES)
            print(
                f"[{label}] Epoch [{epoch+1:3d}/{epochs}]  "
                f"CE={sum_ce/n:.4f}  "
                f"\033[93mTopo={sum_topo/n:.4g}\033[0m  "
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
                "scheduler": scheduler.state_dict(),
                "best_acc":  best_acc,
            }, ckpt_dir / f"best_{label}.pt")

    torch.save({
        "epoch":     epochs - 1,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_acc":  best_acc,
    }, ckpt_dir / f"last_{label}.pt")

    for h in hook_handles:
        h.remove()

    print(f"\n[{label}] Training complete.  Best val acc: {best_acc:.1f}%")

    best_ckpt = torch.load(ckpt_dir / f"best_{label}.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    return model


# -- Main entry ----------------------------------------------------------------

def _make_resnet18(pretrained: bool, device: str):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    return m.to(device)


def _build_topo_loss(model, layer_cfg: dict) -> TopoLoss:
    # topoloss requires nn.Conv2d or nn.Linear leaf layers, not Sequential blocks.
    # We use the last Conv2d of the last BasicBlock in each residual group.
    # Cortical-sparsity hooks stay on the group outputs (see run_training).
    def _leaf_conv(group_name: str) -> nn.Conv2d:
        block_group = getattr(model, group_name)  # nn.Sequential of BasicBlocks
        last_block  = block_group[-1]             # last BasicBlock
        return last_block.conv2                   # final Conv2d in that block

    return TopoLoss(losses=[
        LaplacianPyramid.from_layer(
            model=model, layer=_leaf_conv(name),
            factor_h=layer_cfg[name]["factor_h"],
            factor_w=layer_cfg[name]["factor_w"],
            scale=layer_cfg[name]["topo_scale"],
        )
        for name in TOPO_LAYER_NAMES
    ])


def _make_optimizer_and_scheduler(model, cfg: dict, epochs: int):
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )
    if cfg["lr_schedule"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:  # step
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 60, 80], gamma=0.1
        )
    return optimizer, scheduler


def _load_imagenet_class_names(data_dir: str):
    """Try to read class names from the ImageNet folder structure."""
    try:
        train_root = Path(data_dir) / "train"
        classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
        if len(classes) == 1000:
            return classes
    except Exception:
        pass
    return None   # caller will fall back to numeric indices


class HFImageNetDataset(Dataset):
    """Wraps a HuggingFace ImageNet dataset row into (tensor, label) pairs.

    Expects rows with keys ``image`` (PIL.Image) and ``label`` (int).
    """

    def __init__(self, hf_dataset, transform=None):
        self._ds        = hf_dataset
        self._transform = transform

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        row   = self._ds[idx]
        img   = row["image"].convert("RGB")
        label = int(row["label"])
        if self._transform is not None:
            img = self._transform(img)
        return img, label


def _make_imagenet_datasets(data_dir: str, train_transform, val_transform,
                             hf_repo: str = None,
                             hf_train_size: int = 0,
                             hf_val_size: int = 0):
    """Load ImageNet train/val datasets with automatic fallback.

    Strategy (tried in order):
      1. datasets.ImageFolder(data_dir/train) + datasets.ImageFolder(data_dir/val)
         — standard layout produced by the official ImageNet download script.
      2. HuggingFace ``load_dataset(hf_repo)`` — downloads a subset from the Hub.
         Requires HF_TOKEN / HUGGING_FACE_HUB_TOKEN in the environment.
         hf_train_size / hf_val_size control how many samples to use
         (0 = entire split).

    If both strategies fail the function raises a descriptive FileNotFoundError.
    """
    data_path = Path(data_dir)

    # -- Strategy 1: ImageFolder (train/ + val/ sub-dirs) --------------------
    train_path = data_path / "train"
    val_path   = data_path / "val"
    if train_path.is_dir() and val_path.is_dir():
        print(f"Loading ImageNet via ImageFolder: {data_dir}")
        train_ds = datasets.ImageFolder(str(train_path), transform=train_transform)
        val_ds   = datasets.ImageFolder(str(val_path),   transform=val_transform)
        return train_ds, val_ds

    # -- Strategy 2: HuggingFace Hub -----------------------------------------
    repo = hf_repo or "ILSVRC/imagenet-1k"
    print(
        f"ImageFolder layout not found at '{data_dir}'\n"
        f"  (expected {train_path} and {val_path})\n"
        f"  Falling back to HuggingFace dataset '{repo}' ..."
    )
    try:
        from datasets import load_dataset as hf_load
    except ImportError as ie:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace data loading.\n"
            "Install it with:  pip install datasets"
        ) from ie

    try:
        # Build split strings — HF supports slice notation e.g. "train[:100000]"
        def _split_str(base, n):
            return f"{base}[:{n}]" if n and n > 0 else base

        train_split = _split_str("train",      hf_train_size)
        val_split   = _split_str("validation", hf_val_size)

        print(f"  Downloading split '{train_split}' from {repo} ...")
        hf_train = hf_load(repo, split=train_split, trust_remote_code=True)
        print(f"  Downloading split '{val_split}'   from {repo} ...")
        hf_val   = hf_load(repo, split=val_split,   trust_remote_code=True)

        train_ds = HFImageNetDataset(hf_train, transform=train_transform)
        val_ds   = HFImageNetDataset(hf_val,   transform=val_transform)
        print(
            f"  HF dataset loaded: {len(train_ds):,} train  {len(val_ds):,} val"
        )
        return train_ds, val_ds
    except Exception as e_hf:
        raise FileNotFoundError(
            f"\n\nCould not load ImageNet.\n"
            f"  Strategy 1 (ImageFolder): no train/ + val/ dirs under '{data_dir}'\n"
            f"  Strategy 2 (HuggingFace '{repo}'): {e_hf}\n\n"
            "Fix options:\n"
            "  A) Set DATA_DIR / --data-dir / \"data_dir\" in JSON to a directory\n"
            "     that contains train/ and val/ sub-dirs (ImageFolder layout).\n"
            "  B) Ensure HF_TOKEN is set and the account has access to the repo,\n"
            "     or change \"hf_repo\" in the JSON config to a public dataset.\n"
            "  C) Adjust \"hf_train_size\" / \"hf_val_size\" in the config if the\n"
            "     download is timing out."
        ) from e_hf


def train(cfg: dict):
    print("=" * 65)
    print("  EFFECTIVE CONFIG")
    print("=" * 65)
    print(json.dumps({k: v for k, v in cfg.items()
                      if k not in ("layers", "vis_class_indices")}, indent=2))
    print(f"  vis_class_indices: {cfg['vis_class_indices']}")
    print("  layers:")
    for lname, lvals in cfg["layers"].items():
        print(f"    {lname}: {json.dumps(lvals)}")
    print()

    device   = cfg["device"] if torch.cuda.is_available() else "cpu"
    out_dir  = Path(cfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    use_amp  = cfg["amp"] and torch.cuda.is_available()

    # -- Data ------------------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    data_dir           = cfg["data_dir"]
    train_ds, val_ds   = _make_imagenet_datasets(
        data_dir, train_transform, val_transform,
        hf_repo=cfg.get("hf_repo"),
        hf_train_size=cfg.get("hf_train_size", 0),
        hf_val_size=cfg.get("hf_val_size", 0),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True,
                              persistent_workers=True)
    print(f"Dataset: {len(train_ds):,} train | {len(val_ds):,} val | 1000 classes\n")

    class_names = _load_imagenet_class_names(data_dir)
    vis_cls     = cfg["vis_class_indices"]
    layer_cfg   = cfg["layers"]
    criterion   = nn.CrossEntropyLoss().to(device)

    # -- Topo model ------------------------------------------------------------
    print("=" * 65)
    print("  TOPO MODEL  (TopoLoss + sparsity on layer1…layer4)")
    print("=" * 65)
    topo_model    = _make_resnet18(cfg["pretrained"], device)
    topo_opt, topo_sched = _make_optimizer_and_scheduler(topo_model, cfg, cfg["epochs"])
    topo_scaler   = GradScaler(enabled=use_amp)
    topo_start, topo_best = 0, 0.0

    if cfg.get("resume_topo"):
        ckpt = torch.load(cfg["resume_topo"], map_location=device)
        topo_model.load_state_dict(ckpt["model"])
        topo_opt.load_state_dict(ckpt["optimizer"])
        topo_sched.load_state_dict(ckpt["scheduler"])
        topo_start = ckpt["epoch"] + 1
        topo_best  = ckpt.get("best_acc", 0.0)
        print(f"  Resumed from epoch {topo_start}")

    topo_model = run_training(
        label="topo",
        model=topo_model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=topo_opt, scheduler=topo_sched,
        scaler=topo_scaler, epochs=cfg["epochs"], ckpt_dir=ckpt_dir,
        device=device, print_freq=cfg["print_freq"], use_amp=use_amp,
        topo_loss=_build_topo_loss(topo_model, layer_cfg),
        layer_cfg=layer_cfg,
        start_epoch=topo_start, best_acc=topo_best,
    )

    # -- Topo-only model -------------------------------------------------------
    print("\n" + "=" * 65)
    print("  TOPO-ONLY MODEL  (TopoLoss only, no KL/entropy sparsity)")
    print("=" * 65)
    topo_only_model = _make_resnet18(cfg["pretrained"], device)
    to_opt, to_sched = _make_optimizer_and_scheduler(topo_only_model, cfg, cfg["epochs"])
    to_scaler   = GradScaler(enabled=use_amp)
    to_start, to_best = 0, 0.0

    if cfg.get("resume_topo_only"):
        ckpt = torch.load(cfg["resume_topo_only"], map_location=device)
        topo_only_model.load_state_dict(ckpt["model"])
        to_opt.load_state_dict(ckpt["optimizer"])
        to_sched.load_state_dict(ckpt["scheduler"])
        to_start = ckpt["epoch"] + 1
        to_best  = ckpt.get("best_acc", 0.0)
        print(f"  Resumed from epoch {to_start}")

    topo_only_layer_cfg = {
        name: {**vals, "lambda_kl": 0.0, "lambda_entropy": 0.0}
        for name, vals in layer_cfg.items()
    }
    topo_only_model = run_training(
        label="topo_only",
        model=topo_only_model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=to_opt, scheduler=to_sched,
        scaler=to_scaler, epochs=cfg["epochs"], ckpt_dir=ckpt_dir,
        device=device, print_freq=cfg["print_freq"], use_amp=use_amp,
        topo_loss=_build_topo_loss(topo_only_model, layer_cfg),
        layer_cfg=topo_only_layer_cfg,
        start_epoch=to_start, best_acc=to_best,
    )

    # -- Baseline model --------------------------------------------------------
    print("\n" + "=" * 65)
    print("  BASELINE MODEL  (CE only)")
    print("=" * 65)
    base_model    = _make_resnet18(cfg["pretrained"], device)
    base_opt, base_sched = _make_optimizer_and_scheduler(base_model, cfg, cfg["epochs"])
    base_scaler   = GradScaler(enabled=use_amp)
    base_start, base_best = 0, 0.0

    if cfg.get("resume_base"):
        ckpt = torch.load(cfg["resume_base"], map_location=device)
        base_model.load_state_dict(ckpt["model"])
        base_opt.load_state_dict(ckpt["optimizer"])
        base_sched.load_state_dict(ckpt["scheduler"])
        base_start = ckpt["epoch"] + 1
        base_best  = ckpt.get("best_acc", 0.0)
        print(f"  Resumed from epoch {base_start}")

    base_model = run_training(
        label="baseline",
        model=base_model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=base_opt, scheduler=base_sched,
        scaler=base_scaler, epochs=cfg["epochs"], ckpt_dir=ckpt_dir,
        device=device, print_freq=cfg["print_freq"], use_amp=use_amp,
        topo_loss=None,
        start_epoch=base_start, best_acc=base_best,
    )

    # -- Visualisations --------------------------------------------------------
    all_models = {
        "topo":      topo_model,
        "topo-only": topo_only_model,
        "baseline":  base_model,
    }
    print("\n" + "=" * 65)
    print("  Generating selectivity maps ...")
    print("=" * 65)
    for lbl, mdl in all_models.items():
        save_selectivity_maps(mdl, val_loader, out_dir, device,
                              vis_cls, class_names, tag=lbl)
    save_comparison_figure(all_models, val_loader, out_dir, device,
                           vis_cls, class_names)
    save_activation_cortical_sheets(all_models, val_loader, out_dir, device,
                                    vis_cls, class_names)
    save_debug_cortical_sheets(all_models, val_loader,
                               layer_cfg=layer_cfg,
                               out_dir=out_dir, device=device)

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    cfg = get_config()
    train(cfg)
