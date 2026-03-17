"""Analysis script for the FashionMNIST catastrophic forgetting experiment.

Reads outputs/fashion_mnist_forgetting/results/fmnist_forgetting_results_latest.json
and writes all figures to outputs/fashion_mnist_forgetting/figures/.

Plots
-----
  1_pretrain_val_acc.png        — val accuracy during pretraining (all variants)
  2_finetune_val_acc.png        — val accuracy during finetuning (forgetting curve)
  3_accuracy_overview.png       — 2×2 panel: pretrain acc, finetune acc, forgetting bar, before/after
  4_forgetting_bar.png          — accuracy drop (forgetting) bar chart per variant
  5_per_class_acc.png           — per-class accuracy before vs after (grouped bars)
  6_per_class_forgetting.png    — per-class forgetting (Δacc) per variant
  7_ce_loss.png                 — CE loss, pretrain and finetune side by side
  8_topo_loss.png               — TopoLoss, pretrain and finetune (topo variants only)
  9_kl_loss.png                 — KL penalty, pretrain and finetune (topo_sparsity only)
  10_entropy_loss.png           — Entropy penalty, pretrain and finetune (topo_sparsity only)
  11_grad_entropy.png           — Gradient entropy, pretrain and finetune (all variants)
  12a_auxk_losses.png           — AuxK reconstruction + aux loss (topo_auxk only)
  12b_auxk_dead_frac.png        — AuxK dead latent fraction (topo_auxk only)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Import model + helpers from sibling training module
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from fmnist_forgetting import SimpleNN, SimpleNNAuxK, TOPO_LAYER_NAMES
try:
    from topoloss.core import find_cortical_sheet_size
except ImportError:
    find_cortical_sheet_size = None

# ---------------------------------------------------------------------------
# Constants (must match fmnist_forgetting.py)
# ---------------------------------------------------------------------------

VARIANT_LABELS  = ["baseline", "topo_only", "topo_sparsity", "topo_auxk"]
TOPO_VARIANTS   = ["topo_only", "topo_sparsity", "topo_auxk"]

DISPLAY_NAMES = {
    "baseline":      "Baseline",
    "topo_only":     "Topo Only",
    "topo_sparsity": "Topo + Sparsity",
    "topo_auxk":     "Topo + AuxK",
}

COLORS = {
    "baseline":      "#757575",
    "topo_only":     "#2196f3",
    "topo_sparsity": "#4caf50",
    "topo_auxk":     "#ff9800",
}

MARKERS = {
    "baseline":      "o",
    "topo_only":     "s",
    "topo_sparsity": "^",
    "topo_auxk":     "D",
}

FMNIST_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",  "Shirt",   "Sneaker",  "Bag",   "AnkleBoot",
]

BASE_DIR    = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "outputs" / "fashion_mnist_forgetting" / "results"
FIGURES_DIR = BASE_DIR / "outputs" / "fashion_mnist_forgetting" / "figures"
CKPT_DIR    = BASE_DIR / "outputs" / "fashion_mnist_forgetting" / "checkpoints"

VIS_LAYER_NAMES = ["fc1", "fc2"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(d: dict, *keys):
    """Return the first non-empty list found in d under any of *keys*."""
    for k in keys:
        v = d.get(k)
        if v:
            return v
    return []


def _epochs(arr):
    return list(range(1, len(arr) + 1))


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def _add_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    visible = [(h, l) for h, l in zip(handles, labels) if not l.startswith("_")]
    if visible:
        ax.legend(*zip(*visible), fontsize=9, **kwargs)


def _loss_fmt(ax):
    ax.grid(linestyle="--", alpha=0.4)


# ---------------------------------------------------------------------------
# Two-phase loss plot (pretrain | finetune side-by-side)
# ---------------------------------------------------------------------------

def _two_phase_plot(
    data: dict,
    pretrain_key: str,
    ft_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
    variants=None,
):
    """Paired pretrain / finetune panels, one line per variant."""
    variants = variants or VARIANT_LABELS
    present  = [l for l in variants if l in data]
    if not present:
        print(f"  [SKIP] {out_path.name} — no data")
        return

    fig, (ax_pre, ax_ft) = plt.subplots(1, 2, figsize=(12, 4))

    for label in present:
        curve = _get(data[label], pretrain_key)
        if curve:
            ax_pre.plot(_epochs(curve), curve,
                        label=DISPLAY_NAMES[label], color=COLORS[label],
                        linewidth=1.8, marker=MARKERS[label], markersize=3,
                        markevery=max(1, len(curve) // 15))

    for label in present:
        curve = _get(data[label], ft_key)
        if curve:
            ax_ft.plot(_epochs(curve), curve,
                       label=DISPLAY_NAMES[label], color=COLORS[label],
                       linewidth=1.8, marker=MARKERS[label], markersize=3,
                       markevery=max(1, len(curve) // 15))

    for ax, phase in [(ax_pre, "Pretraining Phase"), (ax_ft, "Noise-Finetuning Phase")]:
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(phase, fontsize=11)
        _loss_fmt(ax)
        _add_legend(ax)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_pretrain_acc(data: dict, out_dir: Path):
    present = [l for l in VARIANT_LABELS if l in data]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for label in present:
        curve = _get(data[label], "pretrain_val_acc_per_epoch")
        if curve:
            ax.plot(_epochs(curve), curve,
                    label=DISPLAY_NAMES[label], color=COLORS[label],
                    linewidth=1.8, marker=MARKERS[label], markersize=3,
                    markevery=max(1, len(curve) // 15))
    ax.set_xlabel("Pretrain Epoch", fontsize=11)
    ax.set_ylabel("Val Accuracy (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_title("FashionMNIST Val Accuracy During Pretraining (all 10 classes)", fontsize=12)
    _add_legend(ax)
    _loss_fmt(ax)
    fig.tight_layout()
    _save(fig, out_dir / "1_pretrain_val_acc.png")


def plot_finetune_acc(data: dict, out_dir: Path):
    present = [l for l in VARIANT_LABELS if l in data]
    if not present:
        return
    # What class is being finetuned?
    noise_class_name = None
    for v in present:
        noise_class_name = data[v].get("noise_target_name")
        if noise_class_name:
            break
    subtitle = f"(noise finetuning: all images labeled as '{noise_class_name}')" if noise_class_name else ""

    fig, ax = plt.subplots(figsize=(8, 4))
    for label in present:
        curve = _get(data[label], "ft_val_acc_per_epoch")
        if curve:
            ax.plot(_epochs(curve), curve,
                    label=DISPLAY_NAMES[label], color=COLORS[label],
                    linewidth=1.8, marker=MARKERS[label], markersize=3,
                    markevery=max(1, len(curve) // 15))
    ax.set_xlabel("Finetune Epoch", fontsize=11)
    ax.set_ylabel("FashionMNIST Val Accuracy (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_title(f"FashionMNIST Val Accuracy During Noise Finetuning\n{subtitle}", fontsize=12)
    _add_legend(ax)
    _loss_fmt(ax)
    fig.tight_layout()
    _save(fig, out_dir / "2_finetune_val_acc.png")


def plot_accuracy_overview(data: dict, out_dir: Path):
    present = [l for l in VARIANT_LABELS if l in data]
    if not present:
        return
    noise_target_name = None
    for v in present:
        noise_target_name = data[v].get("noise_target_name")
        if noise_target_name:
            break

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: pretrain accuracy
    ax = axes[0, 0]
    for label in present:
        curve = _get(data[label], "pretrain_val_acc_per_epoch")
        if curve:
            ax.plot(_epochs(curve), curve, label=DISPLAY_NAMES[label],
                    color=COLORS[label], linewidth=1.8, marker=MARKERS[label],
                    markersize=3, markevery=max(1, len(curve) // 15))
    ax.set_title("Pretrain Val Accuracy", fontsize=11)
    ax.set_xlabel("Pretrain Epoch"); ax.set_ylabel("Val Accuracy (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    _add_legend(ax); _loss_fmt(ax)

    # Top-right: finetune accuracy (forgetting)
    ax = axes[0, 1]
    for label in present:
        curve = _get(data[label], "ft_val_acc_per_epoch")
        if curve:
            ax.plot(_epochs(curve), curve, label=DISPLAY_NAMES[label],
                    color=COLORS[label], linewidth=1.8, marker=MARKERS[label],
                    markersize=3, markevery=max(1, len(curve) // 15))
    ax.set_title("Finetune Val Accuracy (Forgetting)", fontsize=11)
    ax.set_xlabel("Finetune Epoch"); ax.set_ylabel("FashionMNIST Val Accuracy (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    _add_legend(ax); _loss_fmt(ax)

    # Bottom-left: val accuracy before vs after bars
    ax = axes[1, 0]
    x  = np.arange(len(present))
    bw = 0.35
    befores = [data[l].get("val_acc_before", 0) for l in present]
    afters  = [data[l].get("val_acc_after",  0) for l in present]
    ax.bar(x - bw / 2, befores, bw, label="Before finetune",
           color=[COLORS[l] for l in present], alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + bw / 2, afters,  bw, label="After finetune",
           color=[COLORS[l] for l in present], alpha=0.40, edgecolor="black", linewidth=0.5, hatch="///")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[l] for l in present], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Val Accuracy (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_title("Val Accuracy Before vs After Finetuning", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="gray", alpha=0.85, label="Before"),
        Patch(facecolor="gray", alpha=0.40, hatch="///", label="After"),
    ], fontsize=9)
    _loss_fmt(ax)

    # Bottom-right: forgetting (drop)
    ax = axes[1, 1]
    drops = [data[l].get("forgetting_pp", data[l].get("val_acc_before", 0) - data[l].get("val_acc_after", 0))
             for l in present]
    bars  = ax.bar(x, drops, width=0.5,
                   color=[COLORS[l] for l in present], edgecolor="black", linewidth=0.5)
    for bar, drop in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{drop:.1f}pp", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[l] for l in present], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy Drop (pp)")
    ax.set_title("Catastrophic Forgetting (Δ Val Accuracy)", fontsize=11)
    _loss_fmt(ax)

    fig.suptitle(
        f"FashionMNIST Catastrophic Forgetting Overview\n"
        f"(noise finetune target: '{noise_target_name}')",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, out_dir / "3_accuracy_overview.png")


def plot_forgetting_bar(data: dict, out_dir: Path):
    present = [l for l in VARIANT_LABELS if l in data]
    if not present:
        return
    drops = [data[l].get("forgetting_pp", data[l].get("val_acc_before", 0) - data[l].get("val_acc_after", 0))
             for l in present]
    x    = np.arange(len(present))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x, drops, width=0.5,
                  color=[COLORS[l] for l in present], edgecolor="black", linewidth=0.6)
    for bar, drop, label in zip(bars, drops, present):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{drop:.1f}pp", ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=COLORS[label])
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[l] for l in present], fontsize=10)
    ax.set_ylabel("Accuracy Drop (percentage points)", fontsize=11)
    ax.set_title("Catastrophic Forgetting\n(drop in FashionMNIST val accuracy after noise finetuning)",
                 fontsize=12)
    _loss_fmt(ax)
    fig.tight_layout()
    _save(fig, out_dir / "4_forgetting_bar.png")


def plot_per_class_acc(data: dict, out_dir: Path):
    present = [l for l in VARIANT_LABELS if l in data]
    if not present:
        return
    has_pc = any("val_acc_per_class_after" in data[l] for l in present)
    if not has_pc:
        print("  [SKIP] per_class_acc — no per-class data")
        return

    noisy_class_idx = None
    for v in present:
        noisy_class_idx = data[v].get("noise_target_class")
        if noisy_class_idx is not None:
            break

    n_cls = 10
    n_var = len(present)
    x     = np.arange(n_cls)
    bw    = 0.72 / n_var

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_idx, (phase_key, phase_title) in enumerate([
        ("val_acc_per_class_before", "Before Noise Finetuning"),
        ("val_acc_per_class_after",  "After Noise Finetuning"),
    ]):
        ax = axes[ax_idx]
        for vi, label in enumerate(present):
            rec  = data[label].get(phase_key, {})
            vals = [float(rec.get(str(c), rec.get(c, float("nan")))) for c in range(n_cls)]
            offsets = x + (vi - (n_var - 1) / 2.0) * bw
            for xi in range(n_cls):
                hatch = "//" if (noisy_class_idx is not None and xi == noisy_class_idx) else ""
                ax.bar(offsets[xi], vals[xi], bw * 0.9,
                       color=COLORS[label], edgecolor="black", linewidth=0.4,
                       hatch=hatch,
                       label=DISPLAY_NAMES[label] if xi == 0 else "_nolegend_",
                       zorder=3)
        if noisy_class_idx is not None:
            ax.axvspan(noisy_class_idx - 0.5, noisy_class_idx + 0.5,
                       color="#ffe0b2", alpha=0.4, zorder=0)
            ax.text(noisy_class_idx, 106,
                    f"noise\ntarget", ha="center", va="bottom",
                    fontsize=7.5, color="#bf360c", fontstyle="italic")
        ax.set_xticks(x)
        ax.set_xticklabels(FMNIST_CLASSES, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Val Accuracy (%)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.set_ylim(0, 116)
        ax.set_title(phase_title, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if ax_idx == 0:
            _add_legend(ax, loc="lower right")

    noise_name = FMNIST_CLASSES[noisy_class_idx] if noisy_class_idx is not None else "?"
    fig.suptitle(
        f"Per-Class FashionMNIST Val Accuracy — Before vs After Noise Finetuning\n"
        f"(hatched bars = noise-target class '{noise_name}')",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, out_dir / "5_per_class_acc.png")


def plot_per_class_forgetting(data: dict, out_dir: Path):
    present = [l for l in VARIANT_LABELS if l in data]
    if not any("val_acc_per_class_after" in data[l] for l in present):
        print("  [SKIP] per_class_forgetting — no per-class data")
        return

    n_cls = 10
    n_var = len(present)
    x     = np.arange(n_cls)
    bw    = 0.72 / n_var

    noisy_idx = None
    for v in present:
        noisy_idx = data[v].get("noise_target_class")
        if noisy_idx is not None:
            break

    fig, ax = plt.subplots(figsize=(14, 5))
    for vi, label in enumerate(present):
        before = data[label].get("val_acc_per_class_before", {})
        after  = data[label].get("val_acc_per_class_after",  {})
        drops  = []
        for cls in range(n_cls):
            b = float(before.get(str(cls), float("nan")))
            a = float(after.get(str(cls),  float("nan")))
            drops.append(b - a)
        offsets = x + (vi - (n_var - 1) / 2.0) * bw
        ax.bar(offsets, drops, bw * 0.9,
               color=COLORS[label], edgecolor="black", linewidth=0.4,
               label=DISPLAY_NAMES[label], zorder=3)

    if noisy_idx is not None:
        ax.axvspan(noisy_idx - 0.5, noisy_idx + 0.5, color="#ffe0b2", alpha=0.4, zorder=0)
        ax.text(noisy_idx, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 5,
                "noise\ntarget", ha="center", va="top",
                fontsize=7.5, color="#bf360c", fontstyle="italic")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(FMNIST_CLASSES, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy Drop (pp)")
    ax.set_title(
        "Per-Class Catastrophic Forgetting\n"
        "(positive = forgot, negative = improved)",
        fontsize=12,
    )
    _add_legend(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, out_dir / "6_per_class_forgetting.png")


def plot_ce_loss(data: dict, out_dir: Path):
    _two_phase_plot(
        data,
        pretrain_key="pretrain_ce_per_epoch",
        ft_key="ft_ce_per_epoch",
        ylabel="Cross-Entropy Loss",
        title="CE Loss — Pretraining and Noise-Finetuning Phases (all variants)",
        out_path=out_dir / "7_ce_loss.png",
    )


def plot_topo_loss(data: dict, out_dir: Path):
    _two_phase_plot(
        data,
        pretrain_key="pretrain_topo_per_epoch",
        ft_key="ft_topo_per_epoch",
        ylabel="TopoLoss (Laplacian Pyramid)",
        title="Topographic Loss — Pretraining and Noise-Finetuning Phases (topo variants)",
        out_path=out_dir / "8_topo_loss.png",
        variants=TOPO_VARIANTS,
    )


def plot_kl_loss(data: dict, out_dir: Path):
    _two_phase_plot(
        data,
        pretrain_key="pretrain_kl_per_epoch",
        ft_key="ft_kl_per_epoch",
        ylabel="KL Divergence Penalty",
        title="KL-from-Uniform Cortical Sparsity — Pretraining and Finetuning (topo_sparsity only)",
        out_path=out_dir / "9_kl_loss.png",
        variants=["topo_sparsity"],
    )


def plot_entropy_loss(data: dict, out_dir: Path):
    _two_phase_plot(
        data,
        pretrain_key="pretrain_entropy_per_epoch",
        ft_key="ft_entropy_per_epoch",
        ylabel="Per-Sample Entropy Penalty",
        title="Cortical Entropy Penalty — Pretraining and Finetuning (topo_sparsity only)",
        out_path=out_dir / "10_entropy_loss.png",
        variants=["topo_sparsity"],
    )


def plot_grad_entropy(data: dict, out_dir: Path):
    _two_phase_plot(
        data,
        pretrain_key="pretrain_grad_entropy_per_epoch",
        ft_key="ft_grad_entropy_per_epoch",
        ylabel="Gradient Entropy H_a (normalised, 0–1)",
        title="Gradient Entropy — Pretraining and Noise-Finetuning Phases (all variants)",
        out_path=out_dir / "11_grad_entropy.png",
    )


def plot_auxk_losses(data: dict, out_dir: Path):
    """Plot 12a/12b — AuxK reconstruction + aux loss, and dead latent fraction."""
    if "topo_auxk" not in data:
        print("  [SKIP] auxk_losses — no topo_auxk variant")
        return

    d = data["topo_auxk"]
    recon_pre = _get(d, "pretrain_auxk_recon_per_epoch")
    recon_ft  = _get(d, "ft_auxk_recon_per_epoch")
    aux_pre   = _get(d, "pretrain_auxk_aux_per_epoch")
    aux_ft    = _get(d, "ft_auxk_aux_per_epoch")
    dead_pre  = _get(d, "pretrain_auxk_dead_frac_per_epoch")
    dead_ft   = _get(d, "ft_auxk_dead_frac_per_epoch")

    color = COLORS["topo_auxk"]

    # 12a: Reconstruction + Aux loss
    if recon_pre or recon_ft:
        fig, (ax_pre, ax_ft) = plt.subplots(1, 2, figsize=(12, 4))
        for ax, recon, aux, phase in [
            (ax_pre, recon_pre, aux_pre, "Pretraining"),
            (ax_ft,  recon_ft,  aux_ft,  "Noise-Finetuning"),
        ]:
            if recon:
                ax.plot(_epochs(recon), recon, label="Recon loss", color=color,
                        linewidth=1.8, marker="D", markersize=3,
                        markevery=max(1, len(recon) // 15))
            if aux:
                ax.plot(_epochs(aux), aux, label="Aux loss (dead latents)", color=color,
                        linewidth=1.8, linestyle="--", marker="x", markersize=4,
                        markevery=max(1, len(aux) // 15))
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("MSE Loss", fontsize=11)
            ax.set_title(f"{phase} Phase", fontsize=11)
            _add_legend(ax)
            _loss_fmt(ax)
        fig.suptitle("AuxK Losses — Reconstruction + Dead-Latent Auxiliary (Topo + AuxK)", fontsize=12)
        fig.tight_layout()
        _save(fig, out_dir / "12a_auxk_losses.png")

    # 12b: Dead latent fraction
    if dead_pre or dead_ft:
        fig, (ax_pre, ax_ft) = plt.subplots(1, 2, figsize=(12, 4))
        for ax, dead, phase in [
            (ax_pre, dead_pre, "Pretraining"),
            (ax_ft,  dead_ft,  "Noise-Finetuning"),
        ]:
            if dead:
                ax.plot(_epochs(dead), [100 * v for v in dead],
                        label="Dead fraction", color=color,
                        linewidth=1.8, marker="D", markersize=3,
                        markevery=max(1, len(dead) // 15))
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("Dead Latent Fraction (%)", fontsize=11)
            ax.set_title(f"{phase} Phase", fontsize=11)
            ax.set_ylim(-5, 105)
            _add_legend(ax)
            _loss_fmt(ax)
        fig.suptitle("AuxK Dead Latent Fraction — Topo + AuxK Variant", fontsize=12)
        fig.tight_layout()
        _save(fig, out_dir / "12b_auxk_dead_frac.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: Path, device: str, hidden_size: int = 256) -> SimpleNN:
    """Load a SimpleNN (or SimpleNNAuxK) from a checkpoint file."""
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    if "fc1_dec.weight" in state:
        model = SimpleNNAuxK(hidden_size=hidden_size).to(device)
    else:
        model = SimpleNN(hidden_size=hidden_size).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _make_val_loader(data_dir: str, batch_size: int = 256) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    ds = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


def _collect_acts(model: SimpleNN, loader: DataLoader, device: str) -> tuple:
    """Run val set; return ({layer: tensor(N,units)}, labels tensor)."""
    acts_store = {n: [] for n in VIS_LAYER_NAMES}
    lbls_store = []
    handles    = []
    for name in VIS_LAYER_NAMES:
        layer = getattr(model, name)
        def _make_hook(n):
            def _h(_m, _i, out):
                acts_store[n].append(out.detach().cpu())
            return _h
        handles.append(layer.register_forward_hook(_make_hook(name)))
    with torch.no_grad():
        for imgs, lbls in loader:
            model(imgs.to(device))
            lbls_store.append(lbls.cpu())
    for h in handles:
        h.remove()
    return ({n: torch.cat(v) for n, v in acts_store.items()}, torch.cat(lbls_store))


def _t_values(acts: torch.Tensor, labels: torch.Tensor, cls: int) -> torch.Tensor:
    """Per-unit t-statistic: target class vs rest."""
    mask   = labels == cls
    target = acts[mask].float()
    other  = acts[~mask].float()
    if target.size(0) < 2 or other.size(0) < 2:
        return torch.zeros(acts.shape[1])
    n_t, n_o     = target.size(0), other.size(0)
    mu_t, mu_o   = target.mean(0), other.mean(0)
    var_t, var_o = target.var(0, unbiased=True), other.var(0, unbiased=True)
    se = (var_t / n_t + var_o / n_o).clamp(min=1e-12).sqrt()
    return ((mu_t - mu_o) / se).cpu()


def _act_to_sheet(act: torch.Tensor) -> "np.ndarray":
    """Single-sample activation -> 2-D cortical sheet, min-max normalised."""
    a = act.detach().cpu().float().flatten()
    if find_cortical_sheet_size is not None:
        size = find_cortical_sheet_size(a.shape[0])
        a    = a[: size.height * size.width]
        H, W = size.height, size.width
    else:
        side = int(a.shape[0] ** 0.5)
        H = W = side
        a = a[: H * W]
    lo, hi = a.min(), a.max()
    if hi > lo:
        a = (a - lo) / (hi - lo)
    return a.reshape(H, W).numpy()


# ---------------------------------------------------------------------------
# Selectivity diagrams — before vs. after finetuning
# ---------------------------------------------------------------------------

def plot_selectivity_diagrams(ckpt_dir: Path, data_dir: str,
                              out_dir: Path, device: str = "cpu") -> None:
    """Plot 12 — t-statistic selectivity maps for each variant, before and after.

    For each variant two checkpoints are loaded:
      best_pretrain_{variant}.pt   — peak pretrain weights
      last_finetune_{variant}.pt   — post-noise-finetune weights

    Each figure has 10 rows (one per FashionMNIST class) × 5 cols:
      [example image | before/fc1 | before/fc2 | after/fc1 | after/fc2]

    Saved to: out_dir/selectivity/12_selectivity_{variant}.png
    """
    if find_cortical_sheet_size is None:
        print("  [SKIP] selectivity — topoloss not importable")
        return

    sel_dir = out_dir / "selectivity"
    sel_dir.mkdir(parents=True, exist_ok=True)

    loader = _make_val_loader(data_dir)

    # One example image per class
    example_imgs: dict = {}
    for imgs, lbls in loader:
        for cls in range(len(FMNIST_CLASSES)):
            if cls not in example_imgs:
                idx = (lbls == cls).nonzero(as_tuple=True)[0]
                if len(idx):
                    example_imgs[cls] = imgs[idx[0]].squeeze().numpy()
        if len(example_imgs) == len(FMNIST_CLASSES):
            break

    for variant in VARIANT_LABELS:
        pre_path = ckpt_dir / f"best_pretrain_{variant}.pt"
        ft_path  = ckpt_dir / f"last_finetune_{variant}.pt"
        if not pre_path.exists() or not ft_path.exists():
            print(f"  [SKIP] selectivity {variant} — checkpoints not found")
            continue

        print(f"  Selectivity diagrams for '{variant}' ...")
        model_pre = _load_model(pre_path, device)
        model_ft  = _load_model(ft_path,  device)

        acts_pre, labels_pre = _collect_acts(model_pre, loader, device)
        acts_ft,  labels_ft  = _collect_acts(model_ft,  loader, device)

        n_cls  = len(FMNIST_CLASSES)
        # cols: example | fc1_before | fc2_before | fc1_after | fc2_after
        n_cols = 1 + 2 * len(VIS_LAYER_NAMES)
        fig, axes = plt.subplots(
            n_cls, n_cols,
            figsize=(n_cols * 2.4, n_cls * 2.4),
            gridspec_kw={"width_ratios": [1] + [2] * (n_cols - 1)},
        )

        # Column headers
        axes[0, 0].set_title("Example", fontsize=9, pad=3)
        for i, name in enumerate(VIS_LAYER_NAMES):
            axes[0, 1 + i].set_title(f"Before\n{name}", fontsize=8, pad=3)
            axes[0, 1 + len(VIS_LAYER_NAMES) + i].set_title(f"After\n{name}", fontsize=8, pad=3)

        for row, cls in enumerate(range(n_cls)):
            # Input image
            axes[row, 0].imshow(example_imgs.get(cls, np.zeros((28, 28))), cmap="gray")
            axes[row, 0].set_ylabel(FMNIST_CLASSES[cls], fontsize=8, rotation=0,
                                    labelpad=50, va="center")
            axes[row, 0].axis("off")

            for i, name in enumerate(VIS_LAYER_NAMES):
                for phase_idx, (acts, labels) in enumerate([
                    (acts_pre, labels_pre), (acts_ft, labels_ft)
                ]):
                    tv   = _t_values(acts[name], labels, cls)
                    size = find_cortical_sheet_size(tv.shape[0])
                    tmap = tv[: size.height * size.width].reshape(size.height, size.width).numpy()
                    vmax = max(abs(tmap.min()), abs(tmap.max()), 0.1)
                    col  = 1 + phase_idx * len(VIS_LAYER_NAMES) + i
                    im   = axes[row, col].imshow(tmap, cmap="RdGy_r", vmin=-vmax, vmax=vmax)
                    axes[row, col].axis("off")
                    if row == 0:
                        fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04, format="%.1f")

        # Divider between before / after column groups
        ax_r = axes[0, len(VIS_LAYER_NAMES)]
        ax_l = axes[0, len(VIS_LAYER_NAMES) + 1]
        x = (ax_r.get_position().x1 + ax_l.get_position().x0) / 2
        fig.add_artist(plt.Line2D([x, x], [0.02, 0.96],
                                  transform=fig.transFigure,
                                  color="steelblue", linestyle="--", linewidth=1.2,
                                  clip_on=False))

        fig.suptitle(
            f"Category Selectivity — {DISPLAY_NAMES[variant]}\n"
            f"Before (left) vs. After noise finetuning (right)",
            fontsize=12, y=1.002,
        )
        plt.tight_layout()
        path = sel_dir / f"12_selectivity_{variant}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"    Saved -> {path}")


# ---------------------------------------------------------------------------
# Sample activation cortical sheets — before vs. after per variant
# ---------------------------------------------------------------------------

def plot_sample_activations(ckpt_dir: Path, data_dir: str,
                            out_dir: Path, device: str = "cpu") -> None:
    """Plot 13 — one-example-per-class cortical sheet activations, before vs after.

    For each variant: two side-by-side panels (before / after finetuning).
    Rows = FashionMNIST classes, cols = [input | fc1_before | fc2_before | fc1_after | fc2_after].
    Cells are min-max normalised and shown with the 'hot' colormap.

    Saved to: out_dir/activations/13_activations_{variant}.png
    """
    act_dir = out_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    loader = _make_val_loader(data_dir)

    # Collect one example image per class
    example_imgs: dict = {}
    for imgs, lbls in loader:
        for cls in range(len(FMNIST_CLASSES)):
            if cls not in example_imgs:
                idx = (lbls == cls).nonzero(as_tuple=True)[0]
                if len(idx):
                    example_imgs[cls] = imgs[idx[0]:idx[0]+1]   # (1,1,28,28)
        if len(example_imgs) == len(FMNIST_CLASSES):
            break

    def _get_layer_acts(model: SimpleNN, img: torch.Tensor) -> dict:
        store   = {}
        handles = []
        for name in VIS_LAYER_NAMES:
            layer = getattr(model, name)
            def _make_hook(n):
                def _h(_m, _i, out):
                    store[n] = out[0]   # drop batch dim
                return _h
            handles.append(layer.register_forward_hook(_make_hook(name)))
        model.eval()
        with torch.no_grad():
            model(img.to(device))
        for h in handles:
            h.remove()
        return store

    for variant in VARIANT_LABELS:
        pre_path = ckpt_dir / f"best_pretrain_{variant}.pt"
        ft_path  = ckpt_dir / f"last_finetune_{variant}.pt"
        if not pre_path.exists() or not ft_path.exists():
            print(f"  [SKIP] activations {variant} — checkpoints not found")
            continue

        print(f"  Sample activations for '{variant}' ...")
        model_pre = _load_model(pre_path, device)
        model_ft  = _load_model(ft_path,  device)

        n_cls  = len(FMNIST_CLASSES)
        n_lay  = len(VIS_LAYER_NAMES)
        # cols: input | before_fc1 | before_fc2 | after_fc1 | after_fc2
        n_cols = 1 + 2 * n_lay
        fig, axes = plt.subplots(
            n_cls, n_cols,
            figsize=(n_cols * 1.8, n_cls * 1.8),
        )

        # Headers
        axes[0, 0].set_title("Input", fontsize=8, pad=3)
        for i, name in enumerate(VIS_LAYER_NAMES):
            axes[0, 1 + i].set_title(f"Before\n{name}", fontsize=7, pad=3)
            axes[0, 1 + n_lay + i].set_title(f"After\n{name}", fontsize=7, pad=3)

        for row, cls in enumerate(range(n_cls)):
            img_t = example_imgs[cls]
            acts_pre = _get_layer_acts(model_pre, img_t)
            acts_ft  = _get_layer_acts(model_ft,  img_t)

            axes[row, 0].imshow(img_t[0, 0].numpy(), cmap="gray")
            axes[row, 0].set_ylabel(FMNIST_CLASSES[cls], fontsize=7, rotation=0,
                                    labelpad=42, va="center")
            axes[row, 0].axis("off")

            for i, name in enumerate(VIS_LAYER_NAMES):
                for phase_idx, acts in enumerate([acts_pre, acts_ft]):
                    sheet = _act_to_sheet(acts[name])
                    col   = 1 + phase_idx * n_lay + i
                    axes[row, col].imshow(sheet, cmap="hot", vmin=0, vmax=1)
                    axes[row, col].axis("off")

        # Divider between before / after groups
        ax_r = axes[0, n_lay]
        ax_l = axes[0, n_lay + 1]
        x = (ax_r.get_position().x1 + ax_l.get_position().x0) / 2
        fig.add_artist(plt.Line2D([x, x], [0.02, 0.96],
                                  transform=fig.transFigure,
                                  color="steelblue", linestyle="--", linewidth=1.2,
                                  clip_on=False))

        fig.suptitle(
            f"Cortical Sheet Activations — {DISPLAY_NAMES[variant]}\n"
            "Before (left) vs. After noise finetuning (right)  |  min-max normalised per cell",
            fontsize=10, y=1.003,
        )
        plt.tight_layout()
        path = act_dir / f"13_activations_{variant}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"    Saved -> {path}")


def load_data(results_path: Path) -> dict:
    with open(results_path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description="Analyze FashionMNIST forgetting results")
    p.add_argument("--results",  default=None,
                   help="Path to results JSON (default: latest in results dir)")
    p.add_argument("--out-dir",  default=None,
                   help="Output directory for figures (default: outputs/fashion_mnist_forgetting/figures)")
    p.add_argument("--strict",   action="store_true",
                   help="Fail if any variant is missing from the results file")
    p.add_argument("--ckpt-dir",  default=None,
                   help="Checkpoint directory for selectivity/activation plots "
                        "(default: outputs/fashion_mnist_forgetting/checkpoints)")
    p.add_argument("--data-dir",  default=None,
                   help="FashionMNIST data directory (default: project-root/data)")
    p.add_argument("--device",    default="cpu",
                   help="torch device for model forward passes (default: cpu)")
    args = p.parse_args()

    res_dir = RESULTS_DIR
    results_path = Path(args.results)  if args.results  else (res_dir / "fmnist_forgetting_results_latest.json")
    out_dir      = Path(args.out_dir)  if args.out_dir  else FIGURES_DIR
    ckpt_dir     = Path(args.ckpt_dir) if args.ckpt_dir else CKPT_DIR
    BASE_DATA    = BASE_DIR / "data"
    data_dir_str = args.data_dir if args.data_dir else str(BASE_DATA)

    if not results_path.exists():
        print(f"ERROR: results file not found: {results_path}")
        sys.exit(1)

    print(f"Loading results from: {results_path}")
    data = load_data(results_path)

    if args.strict:
        missing = [l for l in VARIANT_LABELS if l not in data]
        if missing:
            print(f"ERROR (--strict): missing variants: {missing}")
            sys.exit(1)

    present = [l for l in VARIANT_LABELS if l in data]
    print(f"Variants found: {present}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Report noise target
    for v in present:
        name = data[v].get("noise_target_name")
        idx  = data[v].get("noise_target_class")
        if name is not None:
            print(f"Noise target: class {idx} ({name})")
            break

    print("\nGenerating accuracy plots …")
    plot_pretrain_acc(data, out_dir)
    plot_finetune_acc(data, out_dir)
    plot_accuracy_overview(data, out_dir)
    plot_forgetting_bar(data, out_dir)

    print("Generating per-class plots …")
    plot_per_class_acc(data, out_dir)
    plot_per_class_forgetting(data, out_dir)

    print("Generating loss curves …")
    plot_ce_loss(data, out_dir)
    plot_topo_loss(data, out_dir)
    plot_kl_loss(data, out_dir)
    plot_entropy_loss(data, out_dir)
    plot_grad_entropy(data, out_dir)
    plot_auxk_losses(data, out_dir)

    print("Generating selectivity diagrams …")
    plot_selectivity_diagrams(ckpt_dir, data_dir_str, out_dir, device=args.device)

    print("Generating sample activation cortical sheets …")
    plot_sample_activations(ckpt_dir, data_dir_str, out_dir, device=args.device)

    print(f"\nAll figures written to: {out_dir}")
    print("\nForgetting summary:")
    for label in present:
        before = data[label].get("val_acc_before", float("nan"))
        after  = data[label].get("val_acc_after",  float("nan"))
        drop   = data[label].get("forgetting_pp",  before - after)
        print(f"  {DISPLAY_NAMES[label]:22s}  before={before:.1f}%  after={after:.1f}%  Δ={drop:+.1f}pp")


if __name__ == "__main__":
    main()
