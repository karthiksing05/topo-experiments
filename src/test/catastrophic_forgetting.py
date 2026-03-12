"""
Catastrophic Forgetting Analysis — STL-10 → CIFAR-10
=====================================================
Loads *_results_latest.json from all three training variants and produces a
full suite of figures and a printed comparison table.

Error checking
--------------
  - Warns clearly if any result file is missing
  - Gracefully skips plots for variants with no data
  - Warns if a required history key is empty / absent in a result file

Plots generated
---------------
  Accuracy
    1_stl_pretrain_acc.png          — STL-10 Val acc per pretrain epoch (all 3)
    2_finetune_cifar_acc.png        — CIFAR-10 Val acc per finetune epoch (all 3)
    3_finetune_stl_acc.png          — STL-10 Val acc per finetune epoch (all 3)
    4_forgetting_bar.png            — Forgetting bar chart (all 3)
    5_accuracy_overview.png         — 2×2 panel summary

  Loss — all variants
    6_pretrain_ce_loss.png          — CE loss per pretrain epoch (all 3)
    7_finetune_ce_loss.png          — CE loss per finetune epoch (all 3)

  Loss — topo variants only
    8_topo_loss_curves.png          — TopoLoss per epoch (pretrain + finetune)
    9_entropy_loss_curves.png       — Entropy loss per epoch (pretrain + finetune,
                                      topo_sparsity only)

  Gradient entropy (all variants)
    10_grad_entropy_curves.png           — Gradient H per epoch (pretrain + finetune, all 3)

  Cortical sheet (requires checkpoints in --ckpt-dir)
    11_cortical_sheets_weights_stl_best.png      — Kernel L2-norm after STL pretrain
    12_cortical_sheets_weights_finetune_last.png — Kernel L2-norm after CIFAR finetune
    13_cortical_sheets_activations_stl_best.png  — Mean activation after STL pretrain
    14_cortical_sheets_activations_finetune_last.png — Mean activation after CIFAR finetune

  Selectivity diagrams (requires checkpoints + data; rows=classes, cols=variants)
    15_selectivity_stl_best_stl10.png            — Post-pretrain ×  STL-10 val stimuli
    16_selectivity_finetune_last_cifar10.png     — Post-finetune × CIFAR-10 val stimuli
    17_selectivity_finetune_last_stl10.png       — Post-finetune ×  STL-10 val stimuli
                                                    (shows representational forgetting)

Usage
-----
  python src/test/catastrophic_forgetting.py \\
      --results-dir outputs/stl_cifar/results \\
      --out-dir     outputs/stl_cifar/figures

  # require all three results to be present (exit 1 otherwise)
  python src/test/catastrophic_forgetting.py --strict

  # include cortical sheet plots (checkpoints + data must be present)
  python src/test/catastrophic_forgetting.py \\
      --ckpt-dir outputs/stl_cifar/checkpoints \\
      --data-dir data/stl_cifar

  # skip cortical sheet plots (e.g. no GPU or checkpoints not yet ready)
  python src/test/catastrophic_forgetting.py --no-cortical-plots

  # skip selectivity diagrams, or change which layer / how many samples
  python src/test/catastrophic_forgetting.py --no-selectivity-plots
  python src/test/catastrophic_forgetting.py --sel-layer layer3.1.conv2 --sel-samples 200
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np

# ── project root ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent

VARIANT_LABELS = ["baseline", "topo_only", "topo_sparsity", "topo_l1sparse", "topo_gradsurg"]
TOPO_VARIANTS  = ["topo_only", "topo_sparsity", "topo_l1sparse", "topo_gradsurg"]

COLORS = {
    "baseline":       "#7f8c8d",
    "topo_only":      "#2980b9",
    "topo_sparsity":  "#27ae60",
    "topo_l1sparse":  "#e67e22",
    "topo_gradsurg":  "#8e44ad",
}
DISPLAY_NAMES = {
    "baseline":      "Baseline",
    "topo_only":     "Topo Only",
    "topo_sparsity": "Topo + Sparsity",
    "topo_l1sparse": "Topo + L1-Sparse",
    "topo_gradsurg": "Topo + GradSurg",
}

STL10_CLASSES  = ["airplane", "bird",   "car",   "cat",   "deer",
                   "dog",     "horse",  "monkey", "ship",  "truck"]
CIFAR10_CLASSES = ["airplane", "automobile", "bird",  "cat",   "deer",
                   "dog",      "frog",       "horse", "ship",  "truck"]

# Default residual conv layer used for selectivity maps (ResNet-18 final block)
SEL_LAYER = "layer4.1.conv2"
MARKERS = {"baseline": "o", "topo_only": "s", "topo_sparsity": "^", "topo_l1sparse": "D", "topo_gradsurg": "P"}


# ─────────────────────────────────────────────────────────────────────────────
# Loading & validation
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_SCALAR_KEYS = [
    "stl_acc_before", "cifar_acc_after", "stl_acc_after", "forgetting",
]


def load_results(results_dir: Path, strict: bool = False) -> dict[str, dict]:
    """Load all variant JSONs; validate keys; report missing/degraded files."""
    data: dict[str, dict] = {}
    errors: list[str] = []
    warnings: list[str] = []

    for label in VARIANT_LABELS:
        path = results_dir / f"{label}_results_latest.json"
        if not path.exists():
            msg = f"Missing result file: {path}"
            errors.append(msg)
            print(f"  [ERROR] {msg}")
            continue

        with open(path) as fh:
            try:
                rec = json.load(fh)
            except json.JSONDecodeError as exc:
                errors.append(f"JSON parse error in {path}: {exc}")
                print(f"  [ERROR] JSON parse error in {path}: {exc}")
                continue

        for key in _REQUIRED_SCALAR_KEYS:
            if key not in rec:
                warnings.append(f"[{label}] missing scalar key '{key}'")

        for key in ["pretrain_ce_per_epoch", "ft_ce_per_epoch"]:
            if key not in rec:
                warnings.append(f"[{label}] missing history key '{key}' "
                                 "(run training script with updated code)")

        if not rec.get("stl_acc_per_epoch"):
            warnings.append(f"[{label}] 'stl_acc_per_epoch' is empty/absent")

        data[label] = rec
        print(f"  Loaded [OK] : {path}")

    for w in warnings:
        print(f"  [WARN]  {w}")

    if strict and errors:
        print(f"\n[FATAL] {len(errors)} missing/invalid result file(s). "
              "Run the training scripts first.")
        sys.exit(1)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(rec: dict, *keys: str, default=None):
    """Try multiple key aliases in order; return first non-empty hit."""
    for k in keys:
        v = rec.get(k)
        if v is not None and (not isinstance(v, list) or len(v) > 0):
            return v
    return default


def _epochs(curve: list) -> range:
    return range(1, len(curve) + 1)


def _apply_acc_fmt(ax):
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))


def _apply_loss_fmt(ax):
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))


def _add_legend(ax, **kwargs):
    ax.legend(fontsize=10, framealpha=0.85, **kwargs)


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Printed table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(data: dict[str, dict]) -> None:
    w = 72
    print()
    print("=" * w)
    print("  STL-10 → CIFAR-10  Catastrophic Forgetting Summary")
    print("=" * w)
    header = (f"  {'Variant':<20} {'STL before':>10} {'CIFAR after':>11} "
              f"{'STL after':>10} {'Forgetting':>11}")
    print(header)
    print("  " + "─" * (w - 2))
    for label in VARIANT_LABELS:
        if label not in data:
            print(f"  {DISPLAY_NAMES[label]:<20}  (no data — run training script)")
            continue
        r = data[label]
        nan = float("nan")
        print(
            f"  {DISPLAY_NAMES[label]:<20}"
            f"  {r.get('stl_acc_before', nan):>8.2f}%"
            f"  {r.get('cifar_acc_after', nan):>9.2f}%"
            f"  {r.get('stl_acc_after', nan):>8.2f}%"
            f"  {r.get('forgetting', nan):>9.2f}%"
        )
    print("=" * w)
    print()
    for label in VARIANT_LABELS:
        if label not in data:
            continue
        cfg = data[label].get("config", {})
        if not cfg:
            continue
        print(f"  [{DISPLAY_NAMES[label]}] config: "
              f"epochs={cfg.get('stl_epochs')}/{cfg.get('finetune_epochs')}  "
              f"lr={cfg.get('lr')}/{cfg.get('finetune_lr')}  "
              f"batch={cfg.get('batch_size')}  img={cfg.get('img_size')}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy plots
# ─────────────────────────────────────────────────────────────────────────────

def _curve_plot(data, key, xlabel, ylabel, title, out_path, figsize=(9, 5)):
    labels_with_data = [l for l in VARIANT_LABELS
                        if l in data and data[l].get(key)]
    if not labels_with_data:
        print(f"  [SKIP] no data for key '{key}'")
        return
    fig, ax = plt.subplots(figsize=figsize)
    for label in labels_with_data:
        curve = data[label][key]
        ax.plot(_epochs(curve), curve,
                label=DISPLAY_NAMES[label], color=COLORS[label],
                linewidth=2, marker=MARKERS[label], markersize=4,
                markevery=max(1, len(curve) // 20))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    _apply_acc_fmt(ax)
    _add_legend(ax)
    fig.tight_layout()
    _save(fig, out_path)


def plot_stl_pretrain_acc(data, out_dir):
    _curve_plot(
        data, key="stl_acc_per_epoch",
        xlabel="Pretrain Epoch", ylabel="STL-10 Val Accuracy (%)",
        title="STL-10 Validation Accuracy During Pretraining",
        out_path=out_dir / "1_stl_pretrain_acc.png",
    )


def plot_finetune_cifar_acc(data, out_dir):
    _curve_plot(
        data, key="cifar_acc_per_epoch_ft",
        xlabel="Finetune Epoch", ylabel="CIFAR-10 Val Accuracy (%)",
        title="CIFAR-10 Validation Accuracy During Finetuning",
        out_path=out_dir / "2_finetune_cifar_acc.png",
    )


def plot_finetune_stl_acc(data, out_dir):
    _curve_plot(
        data, key="stl_acc_per_epoch_ft",
        xlabel="Finetune Epoch", ylabel="STL-10 Val Accuracy (%)",
        title="STL-10 Validation Accuracy During CIFAR-10 Finetuning\n"
              "(drops = catastrophic forgetting)",
        out_path=out_dir / "3_finetune_stl_acc.png",
    )


def plot_forgetting_bar(data, out_dir):
    present = [l for l in VARIANT_LABELS if l in data]
    if not present:
        print("  [SKIP] forgetting bar — no data")
        return
    vals   = [data[l].get("forgetting", 0.0) for l in present]
    colors = [COLORS[l] for l in present]
    names  = [DISPLAY_NAMES[l] for l in present]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.8,
                  zorder=3)
    ax.bar_label(bars, fmt="%.2f%%", padding=5, fontsize=11)
    ax.set_ylabel("Catastrophic Forgetting (STL acc drop, %)", fontsize=12)
    ax.set_title("STL-10 → CIFAR-10: Catastrophic Forgetting per Variant",
                 fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.set_ylim(bottom=0, top=max(vals) * 1.25 if vals else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    fig.tight_layout()
    _save(fig, out_dir / "4_forgetting_bar.png")


def plot_accuracy_overview(data, out_dir):
    """2×2 summary grid: pretrain STL, finetune CIFAR, finetune STL, forgetting."""
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    specs = [
        ("stl_acc_per_epoch",      "Pretrain Epoch",  "STL-10 Accuracy (%)",
         "STL-10 Accuracy — Pretraining", True),
        ("cifar_acc_per_epoch_ft", "Finetune Epoch",  "CIFAR-10 Accuracy (%)",
         "CIFAR-10 Accuracy — Finetuning", True),
        ("stl_acc_per_epoch_ft",   "Finetune Epoch",  "STL-10 Accuracy (%)",
         "STL-10 Accuracy During Finetuning\n(forgetting visible here)", True),
        (None, None, None, "Catastrophic Forgetting Summary", False),
    ]

    for ax, (key, xlabel, ylabel, title, is_line) in zip(axes, specs):
        ax.set_title(title, fontsize=10)
        if is_line:
            for label in VARIANT_LABELS:
                if label not in data or not data[label].get(key):
                    continue
                c = data[label][key]
                ax.plot(_epochs(c), c, label=DISPLAY_NAMES[label],
                        color=COLORS[label], linewidth=1.8,
                        marker=MARKERS[label], markersize=3,
                        markevery=max(1, len(c) // 15))
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            _apply_acc_fmt(ax)
            ax.legend(fontsize=8)
        else:
            present = [l for l in VARIANT_LABELS if l in data]
            vals    = [data[l].get("forgetting", 0.0) for l in present]
            bars = ax.bar(
                [DISPLAY_NAMES[l] for l in present], vals,
                color=[COLORS[l] for l in present],
                edgecolor="black", linewidth=0.7, zorder=3,
            )
            ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
            ax.set_ylabel("Forgetting (%)", fontsize=9)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
            ax.set_ylim(bottom=0, top=max(vals) * 1.3 if vals else 1)
            ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
            ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("STL-10 → CIFAR-10 Catastrophic Forgetting — Overview",
                 fontsize=13, y=1.01)
    _save(fig, out_dir / "5_accuracy_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# Loss plots — all variants
# ─────────────────────────────────────────────────────────────────────────────

def _loss_curve_plot(data, key_aliases, xlabel, ylabel, title, out_path,
                     variants=None, figsize=(9, 5)):
    """Plot a loss curve; tries each alias in key_aliases until data found."""
    if variants is None:
        variants = VARIANT_LABELS
    labels_with_data = [
        l for l in variants
        if l in data and _get(data[l], *key_aliases)
    ]
    if not labels_with_data:
        print(f"  [SKIP] no data for keys {key_aliases}")
        return
    fig, ax = plt.subplots(figsize=figsize)
    for label in labels_with_data:
        curve = _get(data[label], *key_aliases)
        ax.plot(_epochs(curve), curve,
                label=DISPLAY_NAMES[label], color=COLORS[label],
                linewidth=2, marker=MARKERS[label], markersize=3,
                markevery=max(1, len(curve) // 20))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    _apply_loss_fmt(ax)
    _add_legend(ax)
    fig.tight_layout()
    _save(fig, out_path)


def plot_pretrain_ce_loss(data, out_dir):
    _loss_curve_plot(
        data, key_aliases=["pretrain_ce_per_epoch"],
        xlabel="Pretrain Epoch", ylabel="Cross-Entropy Loss",
        title="Cross-Entropy Loss During STL-10 Pretraining (all variants)",
        out_path=out_dir / "6_pretrain_ce_loss.png",
    )


def plot_finetune_ce_loss(data, out_dir):
    _loss_curve_plot(
        data, key_aliases=["ft_ce_per_epoch", "finetune_ce_per_epoch"],
        xlabel="Finetune Epoch", ylabel="Cross-Entropy Loss",
        title="Cross-Entropy Loss During CIFAR-10 Finetuning (all variants)",
        out_path=out_dir / "7_finetune_ce_loss.png",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Loss plots — topo variants only (side-by-side pretrain / finetune panels)
# ─────────────────────────────────────────────────────────────────────────────

def _two_phase_loss_plot(data, pretrain_key, ft_keys, ylabel, title,
                         out_path, variants=None, figsize=(14, 5)):
    """Left panel = pretrain phase; right panel = finetune phase."""
    if variants is None:
        variants = TOPO_VARIANTS

    fig, (ax_pre, ax_ft) = plt.subplots(1, 2, figsize=figsize)
    any_pre = any_ft = False

    for label in variants:
        if label not in data:
            continue
        color, marker = COLORS[label], MARKERS[label]

        pre_curve = _get(data[label], pretrain_key)
        if pre_curve:
            ax_pre.plot(_epochs(pre_curve), pre_curve,
                        label=DISPLAY_NAMES[label],
                        color=color, linewidth=2, marker=marker, markersize=3,
                        markevery=max(1, len(pre_curve) // 20))
            any_pre = True

        ft_curve = _get(data[label], *ft_keys)
        if ft_curve:
            ax_ft.plot(_epochs(ft_curve), ft_curve,
                       label=DISPLAY_NAMES[label],
                       color=color, linewidth=2, marker=marker, markersize=3,
                       markevery=max(1, len(ft_curve) // 15))
            any_ft = True

    if not any_pre and not any_ft:
        print(f"  [SKIP] {title} — no data")
        plt.close(fig)
        return

    for ax, phase, has_data in [
        (ax_pre, "Pretraining", any_pre),
        (ax_ft,  "Finetuning",  any_ft),
    ]:
        ax.set_xlabel(f"{phase} Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{phase} Phase", fontsize=11)
        _apply_loss_fmt(ax)
        if has_data:
            _add_legend(ax)
        else:
            ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_path)


def plot_topo_loss_curves(data, out_dir):
    _two_phase_loss_plot(
        data,
        pretrain_key="pretrain_topo_per_epoch",
        ft_keys=["ft_topo_per_epoch", "finetune_topo_per_epoch"],
        ylabel="TopoLoss",
        title="Topographic Loss (LaplacianPyramid) — Topo Variants Only\n"
               "(GradSurg: real TopoLoss pretrain; flat zero during finetune — gradient surgery replaces topo penalty)",
        out_path=out_dir / "8_topo_loss_curves.png",
    )


def plot_entropy_loss_curves(data, out_dir):
    _two_phase_loss_plot(
        data,
        pretrain_key="pretrain_entropy_per_epoch",
        ft_keys=["ft_entropy_per_epoch", "finetune_entropy_per_epoch"],
        ylabel="Cortical Entropy Loss",
        title="Cortical Entropy (Sparsity) Loss — Topo + Sparsity Only",
        out_path=out_dir / "9_entropy_loss_curves.png",
        variants=["topo_sparsity"],
    )


def plot_l1sparse_loss_curves(data, out_dir):
    """Plot 9b — L1-normalised entropy sparsity loss for the topo_l1sparse variant.

    Minimising this loss pushes raw cortical activations toward exact zeros:
    a silent unit contributes zero probability mass under L1-normalisation,
    so the gradient directly rewards sparsity rather than just shaping a
    softmax distribution.
    """
    _two_phase_loss_plot(
        data,
        pretrain_key="pretrain_sparse_per_epoch",
        ft_keys=["ft_sparse_per_epoch"],
        ylabel="L1-Normalised Entropy Sparsity Loss",
        title="L1-Entropy Sparsity Loss — Topo + L1-Sparse Only",
        out_path=out_dir / "9b_l1sparse_loss_curves.png",
        variants=["topo_l1sparse"],
    )


def plot_grad_entropy_curves(data, out_dir):
    """Plot 10a/10b — gradient entropy across pretrain and finetune phases.

    Gradient entropy (H_grad) is the mean normalised Shannon entropy of the
    per-layer gradient magnitude distribution for each mini-batch step,
    averaged over the step.  Higher → more spatially diffuse gradients;
    lower → more sparse, topographically localised gradient flow.
    All three variants are compared.
    """
    _two_phase_loss_plot(
        data,
        pretrain_key="pretrain_grad_entropy_per_epoch",
        ft_keys=["ft_grad_entropy_per_epoch"],
        ylabel="Gradient Entropy H\u2090 (normalised, 0–1)",
        title="Gradient Entropy Across Training Phases (all variants)",
        out_path=out_dir / "10_grad_entropy_curves.png",
        variants=VARIANT_LABELS,
    )


def plot_kl_curves(data, out_dir):
    """Plot 10b — KL-type penalties: softmax KL (topo_sparsity) + batch-diversity
    L1 SparseKL (topo_l1sparse) overlaid on the same axes.

    - Softmax KL: how far batch-mean softmax activation deviates from uniform.
    - SparseKL: how far batch-mean L1-normalised activation deviates from uniform;
      encourages the cortical sheet to fire across all units at the batch level
      even while individual instances are sparse.
    """
    # Series definitions: (variant, pretrain_key, ft_key, label_suffix, linestyle)
    series = [
        ("topo_sparsity",  "pretrain_kl_per_epoch",        "ft_kl_per_epoch",
         "Softmax KL",  "-"),
        ("topo_l1sparse",  "pretrain_sparse_kl_per_epoch",  "ft_sparse_kl_per_epoch",
         "SparseKL",    "--"),
    ]

    fig, (ax_pre, ax_ft) = plt.subplots(1, 2, figsize=(14, 5))
    any_pre = any_ft = False

    for variant, pre_key, ft_key, suffix, ls in series:
        if variant not in data:
            continue
        color  = COLORS[variant]
        marker = MARKERS[variant]
        name   = f"{DISPLAY_NAMES[variant]} ({suffix})"

        pre_curve = _get(data[variant], pre_key)
        if pre_curve:
            ax_pre.plot(_epochs(pre_curve), pre_curve,
                        label=name, color=color, linewidth=2,
                        linestyle=ls, marker=marker, markersize=3,
                        markevery=max(1, len(pre_curve) // 20))
            any_pre = True

        ft_curve = _get(data[variant], ft_key)
        if ft_curve:
            ax_ft.plot(_epochs(ft_curve), ft_curve,
                       label=name, color=color, linewidth=2,
                       linestyle=ls, marker=marker, markersize=3,
                       markevery=max(1, len(ft_curve) // 15))
            any_ft = True

    if not any_pre and not any_ft:
        print("  [SKIP] plot_kl_curves — no data")
        plt.close(fig)
        return

    for ax, phase, has_data in [
        (ax_pre, "Pretraining", any_pre),
        (ax_ft,  "Finetuning",  any_ft),
    ]:
        ax.set_xlabel(f"{phase} Epoch", fontsize=11)
        ax.set_ylabel("KL Divergence from Uniform", fontsize=11)
        ax.set_title(f"{phase} Phase", fontsize=11)
        _apply_loss_fmt(ax)
        if has_data:
            _add_legend(ax)
        else:
            ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")

    fig.suptitle(
        "Cortical KL Penalties — Softmax KL (topo_sparsity) + SparseKL (topo_l1sparse)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir / "10b_kl_curves.png")


def plot_gradsurg_mask_density(data, out_dir):
    """Plot 10c — gradient-surgery mask density (fraction of cortical units
    NOT zeroed out) during CIFAR-10 finetuning for the topo_gradsurg variant.

    A value of 0.25 means the top-25%% most reactive cortical units per step
    were allowed to update; the remaining 75%% were masked to zero gradient.
    Tracking this confirms the surgery is active and unchanged across epochs.
    """
    label = "topo_gradsurg"
    if label not in data:
        print("  [SKIP] gradsurg_mask_density — topo_gradsurg not in data")
        return
    ft_curve = _get(data[label], "ft_mask_density_per_epoch")
    if not ft_curve:
        print("  [SKIP] gradsurg_mask_density — ft_mask_density_per_epoch absent")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(_epochs(ft_curve), ft_curve,
            color=COLORS[label], linewidth=2,
            marker=MARKERS[label], markersize=4,
            markevery=max(1, len(ft_curve) // 15),
            label=DISPLAY_NAMES[label])
    ax.set_xlabel("Finetuning Epoch", fontsize=11)
    ax.set_ylabel("Mask Density (fraction active)", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.axhline(ft_curve[0] if ft_curve else 0.25, color="gray",
               linewidth=1, linestyle="--", alpha=0.6, label="Expected density")
    ax.set_title(
        "Gradient Surgery — Cortical Mask Density During Finetuning\n"
        "(fraction of units allowed to update per step)",
        fontsize=12,
    )
    _add_legend(ax)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, out_dir / "10c_gradsurg_mask_density.png")


def plot_gradsurg_cortical_entropy(data, out_dir):
    """Plot 10d — Cortical-layer-only gradient entropy for topo_gradsurg.

    Unlike the full-model gradient entropy (Plot 10), which averages over all
    ResNet-18 parameters and is pulled toward 1.0 by the many unmasked backbone
    layers, this plot measures entropy *only* on the cortical conv weights where
    the surgical hooks operate.  Expected behaviour:

      - Pretrain: moderate (~0.93–0.95) — same as other topo variants
      - Finetune: drops sharply toward 0 as the top-25%% mask concentrates
        gradient mass into a small spatial patch each step

    The gap between pretrain and finetune cortical entropy is the cleanest
    diagnostic that gradient surgery is functioning as intended.
    """
    label = "topo_gradsurg"
    if label not in data:
        print("  [SKIP] gradsurg_cortical_entropy — topo_gradsurg not in data")
        return

    pretrain_curve = _get(data[label], "pretrain_grad_entropy_per_epoch")
    ft_curve       = _get(data[label], "ft_cortical_grad_entropy_per_epoch")

    if not pretrain_curve and not ft_curve:
        print("  [SKIP] gradsurg_cortical_entropy — no gradient entropy data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left panel: pretrain full-model entropy (for reference baseline)
    ax = axes[0]
    if pretrain_curve:
        ax.plot(_epochs(pretrain_curve), pretrain_curve,
                color=COLORS[label], linewidth=2,
                marker=MARKERS[label], markersize=3,
                markevery=max(1, len(pretrain_curve) // 15),
                label=DISPLAY_NAMES[label])
    ax.set_xlabel("Pretrain Epoch", fontsize=11)
    ax.set_ylabel("Gradient Entropy H\u2090 (0–1)", fontsize=11)
    ax.set_title("Pretrain Phase — Full-Model Gradient Entropy\n"
                 "(reference: hooks not yet applied)", fontsize=11)
    _add_legend(ax)
    ax.grid(linestyle="--", alpha=0.4)

    # Right panel: finetune cortical-only entropy — the key diagnostic
    ax = axes[1]
    if ft_curve:
        ax.plot(_epochs(ft_curve), ft_curve,
                color=COLORS[label], linewidth=2.5,
                marker=MARKERS[label], markersize=4,
                markevery=max(1, len(ft_curve) // 15),
                label="Cortical layers only")
        # Also show full-model finetune entropy for comparison
        ft_full = _get(data[label], "ft_grad_entropy_per_epoch")
        if ft_full:
            ax.plot(_epochs(ft_full), ft_full,
                    color=COLORS[label], linewidth=1.5, linestyle="--",
                    alpha=0.55,
                    marker=MARKERS[label], markersize=3,
                    markevery=max(1, len(ft_full) // 15),
                    label="All layers (diluted)")
    else:
        ax.text(0.5, 0.5,
                "No ft_cortical_grad_entropy data\n(re-run training to generate)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("Finetune Epoch", fontsize=11)
    ax.set_ylabel("Gradient Entropy H\u2090 (0–1)", fontsize=11)
    ax.set_title("Finetune Phase — Cortical-Only vs. Full-Model Entropy\n"
                 "(cortical entropy shows true sparsification effect)", fontsize=11)
    _add_legend(ax)
    ax.grid(linestyle="--", alpha=0.4)

    fig.suptitle(
        "Gradient Surgery — Cortical-Layer Gradient Entropy\n"
        "(cortical-only metric isolates the surgical masking effect from backbone dilution)",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, out_dir / "10d_gradsurg_cortical_entropy.png")


def plot_gradsurg_finetune_comparison(data, out_dir):
    """Plot 11 — Gradient-surgery dedicated finetune comparison.

    Four-panel figure isolating what makes the finetune phase for topo_gradsurg
    distinctive vs. the loss-penalty based topo variants:

      Top-left:  Finetune CE loss (all variants) — gradsurg: CE only, no
                 auxiliary penalties; baseline has no regularisation either, so
                 the comparison shows whether gated gradients hurt CE optimisation.
      Top-right: Gradient entropy during finetuning only (all topo variants).
                 Key story: masked gradients → lower entropy for gradsurg.
      Bottom-left: STL-10 accuracy trajectory during CIFAR-10 finetuning.
                 Shows how quickly STL-10 performance degrades for each variant.
      Bottom-right: Pretrain TopoLoss (topo variants only) — confirms gradsurg
                 has the same pretrain objective as the other topo methods, making
                 the finetune difference a fair comparison of regularisation strategy.
    """
    present = [l for l in VARIANT_LABELS if l in data]
    topo_present = [l for l in TOPO_VARIANTS if l in data]
    if not present:
        print("  [SKIP] gradsurg_finetune_comparison — no variant data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- Top-left: finetune CE loss (all variants) ----------------------------
    ax = axes[0, 0]
    gs_lw = {"topo_gradsurg": 3.0}  # thicker line for gradsurg
    for label in present:
        curve = _get(data[label], "ft_ce_per_epoch", "finetune_ce_per_epoch")
        if not curve:
            continue
        lw = gs_lw.get(label, 1.8)
        ax.plot(_epochs(curve), curve,
                label=DISPLAY_NAMES[label], color=COLORS[label],
                linewidth=lw, marker=MARKERS[label], markersize=3,
                markevery=max(1, len(curve) // 15),
                zorder=3 if label == "topo_gradsurg" else 2)
    ax.set_xlabel("Finetune Epoch", fontsize=11)
    ax.set_ylabel("CE Loss", fontsize=11)
    ax.set_title("Finetune CE Loss — All Variants\n"
                 "(GradSurg: CE only, zero auxiliary penalties)",
                 fontsize=11)
    _add_legend(ax)
    ax.grid(linestyle="--", alpha=0.4)

    # ---- Top-right: gradient entropy finetune only (topo variants) ------------
    ax = axes[0, 1]
    for label in topo_present:
        curve = _get(data[label], "ft_grad_entropy_per_epoch")
        if not curve:
            continue
        lw = 3.0 if label == "topo_gradsurg" else 1.8
        ax.plot(_epochs(curve), curve,
                label=DISPLAY_NAMES[label], color=COLORS[label],
                linewidth=lw, marker=MARKERS[label], markersize=3,
                markevery=max(1, len(curve) // 15),
                zorder=3 if label == "topo_gradsurg" else 2)
    if not any(_get(data[l], "ft_grad_entropy_per_epoch") for l in topo_present):
        ax.text(0.5, 0.5, "No gradient entropy data",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("Finetune Epoch", fontsize=11)
    ax.set_ylabel("Gradient Entropy H\u2090 (0–1)", fontsize=11)
    ax.set_title("Gradient Entropy During Finetuning — Topo Variants\n"
                 "(GradSurg: masked gradients \u2192 lower entropy expected)",
                 fontsize=11)
    _add_legend(ax)
    ax.grid(linestyle="--", alpha=0.4)

    # ---- Bottom-left: STL-10 accuracy during finetuning (all variants) --------
    ax = axes[1, 0]
    for label in present:
        curve = _get(data[label], "ft_stl_acc_per_epoch", "finetune_stl_acc_per_epoch")
        if not curve:
            continue
        lw = 3.0 if label == "topo_gradsurg" else 1.8
        ax.plot(_epochs(curve), curve,
                label=DISPLAY_NAMES[label], color=COLORS[label],
                linewidth=lw, marker=MARKERS[label], markersize=3,
                markevery=max(1, len(curve) // 15),
                zorder=3 if label == "topo_gradsurg" else 2)
    if not any(_get(data[l], "ft_stl_acc_per_epoch", "finetune_stl_acc_per_epoch")
               for l in present):
        ax.text(0.5, 0.5, "No STL-10 finetune accuracy data",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("Finetune Epoch", fontsize=11)
    ax.set_ylabel("STL-10 Val Accuracy (%)", fontsize=11)
    ax.set_title("STL-10 Accuracy During CIFAR-10 Finetuning\n"
                 "(forgetting dynamics — GradSurg highlighted)",
                 fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    _add_legend(ax)
    ax.grid(linestyle="--", alpha=0.4)

    # ---- Bottom-right: pretrain Topo loss — topo variants only ----------------
    ax = axes[1, 1]
    for label in topo_present:
        curve = _get(data[label], "pretrain_topo_per_epoch")
        if not curve:
            continue
        lw = 3.0 if label == "topo_gradsurg" else 1.8
        ax.plot(_epochs(curve), curve,
                label=DISPLAY_NAMES[label], color=COLORS[label],
                linewidth=lw, marker=MARKERS[label], markersize=3,
                markevery=max(1, len(curve) // 15),
                zorder=3 if label == "topo_gradsurg" else 2)
    if not any(_get(data[l], "pretrain_topo_per_epoch") for l in topo_present):
        ax.text(0.5, 0.5, "No pretrain TopoLoss data",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("Pretrain Epoch", fontsize=11)
    ax.set_ylabel("TopoLoss", fontsize=11)
    ax.set_title("Pretrain Topo Loss — Topo Variants\n"
                 "(all topo methods share same pretrain objective)",
                 fontsize=11)
    _add_legend(ax)
    ax.grid(linestyle="--", alpha=0.4)

    fig.suptitle(
        "Gradient Surgery vs. Loss-Penalty Topo Variants — Finetune Phase Breakdown\n"
        "(GradSurg replaces auxiliary loss penalties with gradient masking during CIFAR-10 finetuning)",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, out_dir / "11_gradsurg_finetune_comparison.png")


def plot_stl_per_class_acc(data, out_dir):
    """Plot 18 — Per-class STL-10 accuracy before and after CIFAR-10 finetuning.

    Each STL-10 class gets a group of bars (one per variant).  The STL-10-
    exclusive class (monkey, index 7) is cross-hatched because it has no
    CIFAR-10 counterpart and therefore receives *no* finetuning signal,
    making it the purest probe of catastrophic forgetting.
    """
    STL_EXCLUSIVE_IDX = {7}   # monkey — not in CIFAR-10
    present = [l for l in VARIANT_LABELS if l in data]
    if not present:
        print("  [SKIP] stl_per_class_acc — no data")
        return
    has_pc = any("stl_per_class_acc_after" in data[l] for l in present)
    if not has_pc:
        print("  [SKIP] stl_per_class_acc — per-class data absent "
              "(re-run training to generate)")
        return

    n_cls = 10
    n_var = len(present)
    x     = np.arange(n_cls)
    bar_w = 0.72 / n_var

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_idx, (phase_key, phase_title) in enumerate([
        ("stl_per_class_acc_before", "Post-Pretrain  (Before Finetuning)"),
        ("stl_per_class_acc_after",  "Post-Finetune  (After CIFAR-10 Training)"),
    ]):
        ax = axes[ax_idx]
        for vi, label in enumerate(present):
            rec  = data[label].get(phase_key, {})
            vals = [
                float(rec.get(str(c), rec.get(c, float("nan"))))
                for c in range(n_cls)
            ]
            offsets = x + (vi - (n_var - 1) / 2.0) * bar_w
            for xi in range(n_cls):
                ax.bar(
                    offsets[xi], vals[xi], bar_w * 0.92,
                    color=COLORS[label], edgecolor="black", linewidth=0.5,
                    hatch=("///" if xi in STL_EXCLUSIVE_IDX else ""),
                    label=DISPLAY_NAMES[label] if xi == 0 else "_nolegend_",
                    zorder=3,
                )
                # Annotate monkey bars in the after-finetuning panel with values
                if ax_idx == 1 and xi in STL_EXCLUSIVE_IDX:
                    v = vals[xi]
                    if not (v != v):  # not NaN
                        bar_x = offsets[xi]
                        ax.text(bar_x, v + 0.8, f"{v:.1f}%",
                                ha="center", va="bottom", fontsize=7,
                                color=COLORS[label], fontweight="bold",
                                rotation=90, zorder=5)
        for exc in STL_EXCLUSIVE_IDX:
            ax.axvspan(exc - 0.5, exc + 0.5, color="#ede7f6", alpha=0.5, zorder=0)
            ax.text(exc, 104, "STL-only\n(monkey)", ha="center", va="bottom",
                    fontsize=7.5, color="#6a0dad", fontstyle="italic")
        ax.set_xticks(x)
        ax.set_xticklabels(STL10_CLASSES, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("STL-10 Val Accuracy (%)", fontsize=11)
        ax.set_title(phase_title, fontsize=11)
        ax.set_ylim(0, 116)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        if ax_idx == 0:
            ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        "STL-10 Per-Class Accuracy: Before vs. After CIFAR-10 Finetuning\n"
        "(hatched bars = STL-10 exclusive class absent from CIFAR-10 training)",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, out_dir / "18_stl_per_class_acc.png")


def plot_stl_per_class_forgetting(data, out_dir):
    """Plot 19 — Per-class STL-10 forgetting (acc_before − acc_after) per variant.

    Each STL-10 class gets a group of bars showing how much accuracy was lost
    after CIFAR-10 finetuning.  The STL-exclusive class (monkey, index 7) is
    cross-hatched and its column is highlighted: because the model never sees
    monkey during finetuning, any forgetting there is pure interference from
    the CIFAR task rather than direct overwriting of a shared representation.
    """
    STL_EXCLUSIVE_IDX = {7}   # monkey — absent from CIFAR-10
    present = [l for l in VARIANT_LABELS if l in data]
    if not present:
        print("  [SKIP] stl_per_class_forgetting — no data")
        return
    has_pc = any(
        "stl_per_class_acc_before" in data[l] and "stl_per_class_acc_after" in data[l]
        for l in present
    )
    if not has_pc:
        print("  [SKIP] stl_per_class_forgetting — per-class data absent "
              "(re-run training to generate)")
        return

    n_cls = 10
    n_var = len(present)
    x     = np.arange(n_cls)
    bar_w = 0.72 / n_var

    fig, ax = plt.subplots(figsize=(14, 5))

    for vi, label in enumerate(present):
        rec_before = data[label].get("stl_per_class_acc_before", {})
        rec_after  = data[label].get("stl_per_class_acc_after",  {})
        forgetting = []
        for c in range(n_cls):
            b = float(rec_before.get(str(c), rec_before.get(c, float("nan"))))
            a = float(rec_after.get( str(c), rec_after.get( c, float("nan"))))
            forgetting.append(b - a)   # positive = forgotten, negative = improved
        offsets = x + (vi - (n_var - 1) / 2.0) * bar_w
        for xi in range(n_cls):
            ax.bar(
                offsets[xi], forgetting[xi], bar_w * 0.92,
                color=COLORS[label], edgecolor="black", linewidth=0.5,
                hatch=("///" if xi in STL_EXCLUSIVE_IDX else ""),
                label=DISPLAY_NAMES[label] if xi == 0 else "_nolegend_",
                zorder=3,
            )

    # Highlight the STL-exclusive column — annotate after bars are drawn
    for exc in STL_EXCLUSIVE_IDX:
        ax.axvspan(exc - 0.5, exc + 0.5, color="#ede7f6", alpha=0.45, zorder=0)

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(STL10_CLASSES, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Acc drop after CIFAR finetuning (%)", fontsize=11)
    ax.set_title(
        "STL-10 Per-Class Forgetting After CIFAR-10 Finetuning\n"
        "(positive = forgotten; hatched bar = STL-exclusive class)",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper right")
    # Add annotation after ylim is determined by data
    ymax = ax.get_ylim()[1]
    for exc in STL_EXCLUSIVE_IDX:
        ax.text(exc, ymax * 0.97, "STL-only\n(monkey)", ha="center", va="top",
                fontsize=8, color="#6a0dad", fontstyle="italic")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    _save(fig, out_dir / "19_stl_per_class_forgetting.png")


def plot_finetuned_vs_unfinetuned_forgetting(data, out_dir):
    """Plot 20 — Forgetting split by finetuned / non-finetuned / monkey categories.

    Left panel: per-class forgetting bars identical to plot 19, but with
    background shading marking which STL-10 classes were included in the
    CIFAR-10 finetune subset (green), which were excluded (orange), and the
    STL-exclusive monkey class (purple).

    Right panel: mean forgetting per group per variant as a grouped bar chart,
    quantifying whether non-finetuned classes are better retained.
    """
    STL_EXCLUSIVE_IDX = {7}   # monkey — absent from CIFAR-10

    present = [l for l in VARIANT_LABELS if l in data]
    if not present:
        print("  [SKIP] finetuned_vs_unfinetuned_forgetting — no data")
        return
    has_pc = any(
        "stl_per_class_acc_before" in data[l] and "stl_per_class_acc_after" in data[l]
        for l in present
    )
    if not has_pc:
        print("  [SKIP] finetuned_vs_unfinetuned_forgetting — per-class data absent")
        return

    # Determine which STL classes were finetuned — read from first available result.
    # Fall back to all 9 CIFAR-overlapping STL classes if key absent (old runs).
    _ALL_OVERLAP_STL = {0, 1, 2, 3, 4, 5, 6, 8, 9}  # classes 0–9 minus monkey (7)
    finetuned_set = None
    for lbl in present:
        ft_classes = data[lbl].get("finetuned_stl_classes")
        if ft_classes is not None:
            finetuned_set = set(ft_classes)
            break
    if finetuned_set is None:
        finetuned_set = _ALL_OVERLAP_STL   # legacy: all overlap classes

    not_finetuned_set = _ALL_OVERLAP_STL - finetuned_set

    n_cls = 10
    n_var = len(present)
    x     = np.arange(n_cls)
    bar_w = 0.72 / n_var

    fig, (ax_pc, ax_grp) = plt.subplots(1, 2, figsize=(18, 6),
                                         gridspec_kw={"width_ratios": [2, 1]})

    # ── Left panel: per-class forgetting with background shading ──────────────
    for vi, label in enumerate(present):
        rec_before = data[label].get("stl_per_class_acc_before", {})
        rec_after  = data[label].get("stl_per_class_acc_after",  {})
        forgetting = []
        for c in range(n_cls):
            b = float(rec_before.get(str(c), rec_before.get(c, float("nan"))))
            a = float(rec_after.get( str(c), rec_after.get( c, float("nan"))))
            forgetting.append(b - a)
        offsets = x + (vi - (n_var - 1) / 2.0) * bar_w
        for xi in range(n_cls):
            ax_pc.bar(
                offsets[xi], forgetting[xi], bar_w * 0.92,
                color=COLORS[label], edgecolor="black", linewidth=0.5,
                hatch=("///" if xi in STL_EXCLUSIVE_IDX else ""),
                label=DISPLAY_NAMES[label] if xi == 0 else "_nolegend_",
                zorder=3,
            )

    # Background shading by group
    for c in range(n_cls):
        if c in STL_EXCLUSIVE_IDX:
            ax_pc.axvspan(c - 0.5, c + 0.5, color="#ede7f6", alpha=0.5, zorder=0)
        elif c in finetuned_set:
            ax_pc.axvspan(c - 0.5, c + 0.5, color="#e8f5e9", alpha=0.5, zorder=0)
        else:
            ax_pc.axvspan(c - 0.5, c + 0.5, color="#fff3e0", alpha=0.5, zorder=0)

    ax_pc.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax_pc.set_xticks(x)
    ax_pc.set_xticklabels(STL10_CLASSES, rotation=30, ha="right", fontsize=9)
    ax_pc.set_ylabel("Acc drop after CIFAR finetuning (%)", fontsize=11)
    ax_pc.set_title(
        "Per-Class STL-10 Forgetting\n"
        "(green = finetuned class; orange = excluded; purple = STL-only)",
        fontsize=11,
    )
    ax_pc.legend(fontsize=8, loc="upper right")
    ax_pc.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # Annotations: group labels above the axis
    ymax_pc = ax_pc.get_ylim()[1]
    for c in range(n_cls):
        if c in STL_EXCLUSIVE_IDX:
            ax_pc.text(c, ymax_pc * 0.97, "STL\nonly", ha="center", va="top",
                       fontsize=7, color="#6a0dad", fontstyle="italic")
        elif c in finetuned_set:
            ax_pc.text(c, ymax_pc * 0.97, "FT", ha="center", va="top",
                       fontsize=7, color="#2e7d32", fontweight="bold")
        else:
            ax_pc.text(c, ymax_pc * 0.97, "excl.", ha="center", va="top",
                       fontsize=7, color="#e65100", fontstyle="italic")

    # ── Right panel: mean forgetting per group per variant ────────────────────
    groups = [
        ("Finetuned",     finetuned_set,      "#4caf50"),
        ("Not Finetuned", not_finetuned_set,  "#ff9800"),
        ("Monkey\n(STL-only)", STL_EXCLUSIVE_IDX, "#9c27b0"),
    ]
    n_grp = len(groups)
    gx    = np.arange(n_grp)
    grp_bar_w = 0.7 / n_var

    for vi, label in enumerate(present):
        rec_before = data[label].get("stl_per_class_acc_before", {})
        rec_after  = data[label].get("stl_per_class_acc_after",  {})
        means = []
        for _, idx_set, _ in groups:
            vals = []
            for c in idx_set:
                b = float(rec_before.get(str(c), rec_before.get(c, float("nan"))))
                a = float(rec_after.get( str(c), rec_after.get( c, float("nan"))))
                if not (np.isnan(b) or np.isnan(a)):
                    vals.append(b - a)
            means.append(float(np.mean(vals)) if vals else float("nan"))

        offsets = gx + (vi - (n_var - 1) / 2.0) * grp_bar_w
        ax_grp.bar(
            offsets, means, grp_bar_w * 0.88,
            color=COLORS[label], edgecolor="black", linewidth=0.5,
            label=DISPLAY_NAMES[label], zorder=3,
        )

    ax_grp.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax_grp.set_xticks(gx)
    ax_grp.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax_grp.set_ylabel("Mean acc drop (%)", fontsize=11)
    ax_grp.set_title("Mean Forgetting by Group", fontsize=11)
    ax_grp.legend(fontsize=8, loc="upper right")
    ax_grp.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    fig.suptitle(
        "STL-10 Forgetting: Finetuned vs Non-Finetuned Categories",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir / "20_finetuned_vs_unfinetuned_forgetting.png")


# ─────────────────────────────────────────────────────────────────────────────
# Text summary
# ─────────────────────────────────────────────────────────────────────────────

def save_text_summary(data: dict, out_dir: Path) -> None:
    out_path = out_dir / "summary.txt"
    lines: list[str] = [
        "STL-10 → CIFAR-10 Catastrophic Forgetting — Results Summary",
        "=" * 68,
        f"{'Variant':<20} {'STL before':>10} {'CIFAR after':>11} "
        f"{'STL after':>10} {'Forgetting':>11}",
        "─" * 68,
    ]
    for label in VARIANT_LABELS:
        if label not in data:
            lines.append(f"{DISPLAY_NAMES[label]:<20}  (no data)")
            continue
        r   = data[label]
        nan = float("nan")
        lines.append(
            f"{DISPLAY_NAMES[label]:<20}"
            f"  {r.get('stl_acc_before', nan):>8.2f}%"
            f"  {r.get('cifar_acc_after', nan):>9.2f}%"
            f"  {r.get('stl_acc_after', nan):>8.2f}%"
            f"  {r.get('forgetting', nan):>9.2f}%"
        )
    lines.append("=" * 68)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Cortical sheet visualizations
# ─────────────────────────────────────────────────────────────────────────────

def _requires_torch():
    """Return the torch module if importable, else None."""
    try:
        import torch
        return torch
    except ImportError:
        return None


def _setup_model_paths():
    """Ensure src/stl_cifar and src/imagenet are on sys.path."""
    for sub in ("stl_cifar", "imagenet"):
        p = str(BASE_DIR / "src" / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_model_for_viz(ckpt_path: Path, device: str = "cpu"):
    """Load a ResNet-18 (10-class) from a checkpoint file, eval mode."""
    import torch
    _setup_model_paths()
    from stl_cifar_common import build_model
    model = build_model(num_classes=10, device=device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _weight_cortical_sheets(model) -> dict:
    """
    For each residual conv in *model*, compute the L2-norm of each
    output-channel's flattened kernel and reshape it onto the (H, W)
    cortical sheet given by find_cortical_sheet_size.

    Returns {layer_name: np.ndarray of shape (H, W)}.
    """
    _setup_model_paths()
    from resnet_imagenet_common import get_residual_convs
    from topoloss.cortical_sheet.output import get_cortical_sheet_conv

    conv_dict = get_residual_convs(model, include_downsample=False)
    sheets: dict = {}
    for name, layer in conv_dict.items():
        try:
            # cs: (H, W, in_channels*kH*kW)
            cs = get_cortical_sheet_conv(layer, strict_layer_type=True)
            sheets[name] = cs.detach().cpu().float().norm(dim=-1).numpy()
        except Exception as exc:
            print(f"  [cortical_sheets] Skipping layer {name}: {exc}")
    return sheets


def _activation_cortical_sheets(model, loader, device: str = "cpu") -> dict:
    """
    Register forward hooks on every residual conv, run one batch from
    *loader*, then reshape the mean per-channel activation onto each
    layer's cortical sheet.

    Returns {layer_name: np.ndarray of shape (H, W)}.
    """
    import torch
    _setup_model_paths()
    from resnet_imagenet_common import get_residual_convs
    from topoloss.cortical_sheet.common import find_cortical_sheet_size

    conv_dict = get_residual_convs(model, include_downsample=False)
    raw_acts: dict = {}
    hooks = []

    def _make_hook(name):
        def _hook(module, inp, out):
            x = out.detach().cpu().float()
            if x.dim() == 4:
                raw_acts[name] = x.mean(dim=(0, 2, 3)).numpy()   # (C,)
            elif x.dim() == 2:
                raw_acts[name] = x.mean(dim=0).numpy()           # (C,)
        return _hook

    for name, layer in conv_dict.items():
        hooks.append(layer.register_forward_hook(_make_hook(name)))

    try:
        imgs, _ = next(iter(loader))
        with torch.no_grad():
            model(imgs[:32].to(device))
    finally:
        for h in hooks:
            h.remove()

    sheets: dict = {}
    for name, act in raw_acts.items():
        C    = act.shape[0]
        size = find_cortical_sheet_size(C)
        sheets[name] = act[: size.height * size.width].reshape(
            size.height, size.width
        )
    return sheets


def _plot_sheet_grid(
    variant_sheets: dict,
    layer_names: list,
    labels_ordered: list,
    title: str,
    out_dir: Path,
    filename: str,
):
    """
    Draw a grid of cortical-sheet heatmaps.
    Rows = conv layers,  Columns = training variants.
    variant_sheets: {label: {layer_name: np.ndarray(H, W)}}
    """
    n_layers   = len(layer_names)
    n_variants = len(labels_ordered)

    fig, axes = plt.subplots(
        n_layers, n_variants,
        figsize=(4 * n_variants, 3 * n_layers),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=13, y=1.01)

    for col, label in enumerate(labels_ordered):
        for row, lname in enumerate(layer_names):
            ax    = axes[row][col]
            sheet = variant_sheets.get(label, {}).get(lname)
            if sheet is None:
                ax.set_visible(False)
                continue
            im = ax.imshow(sheet, cmap="viridis", aspect="auto",
                           interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(label.replace("_", " "), fontsize=10,
                             fontweight="bold")
            if col == 0:
                # shorten e.g. "layer1.0.conv1" → "0.conv1"
                parts = lname.split(".")
                short = ".".join(parts[-2:]) if len(parts) >= 2 else lname
                ax.set_ylabel(short, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    _save(fig, out_dir / filename)


def plot_cortical_sheets_weights(data, ckpt_dir: Path, out_dir: Path):
    """
    Plots 11 & 12 — Weight L2-norm cortical sheets.

    For each checkpoint phase (post-STL pretrain, post-CIFAR finetune) and
    each variant, visualise the cortical sheet of kernel norms for every
    residual conv layer.  No data loader required.
    """
    torch = _requires_torch()
    if torch is None:
        print("  [cortical_sheets] PyTorch not available — skipping weight plots.")
        return

    for plot_idx, stem, phase_label in [
        (11, "stl_best",       "Post-STL Pretrain"),
        (12, "finetune_last",  "Post-CIFAR Finetune"),
    ]:
        variant_sheets:   dict = {}
        layer_names_ord = None

        for label in VARIANT_LABELS:
            ckpt = ckpt_dir / f"{stem}_{label}.pt"
            if not ckpt.exists():
                print(f"  [cortical_sheets] Checkpoint not found: {ckpt.name}")
                continue
            try:
                model  = _load_model_for_viz(ckpt)
                sheets = _weight_cortical_sheets(model)
            except Exception as exc:
                print(f"  [cortical_sheets] {label} ({stem}): {exc}")
                continue
            if sheets:
                variant_sheets[label] = sheets
                if layer_names_ord is None:
                    layer_names_ord = list(sheets.keys())

        if not variant_sheets or not layer_names_ord:
            print(f"  [cortical_sheets] Nothing to plot for {phase_label}.")
            continue

        labels_ord = [l for l in VARIANT_LABELS if l in variant_sheets]
        _plot_sheet_grid(
            variant_sheets, layer_names_ord, labels_ord,
            title=f"Cortical Sheet — Kernel L2-Norm ({phase_label})",
            out_dir=out_dir,
            filename=f"{plot_idx}_cortical_sheets_weights_{stem}.png",
        )


def plot_cortical_sheets_activations(data, ckpt_dir: Path, data_dir: Path,
                                     out_dir: Path):
    """
    Plots 13 & 14 — Activation cortical sheets.

    Runs one STL-10 validation batch through each checkpoint and reshapes
    the mean per-channel activation onto the cortical sheet.
    """
    torch = _requires_torch()
    if torch is None:
        print("  [cortical_sheets] PyTorch not available — skipping activation plots.")
        return

    _setup_model_paths()
    try:
        from stl_cifar_common import make_stl_loaders
    except ImportError as exc:
        print(f"  [cortical_sheets] Cannot import stl_cifar_common: {exc}")
        return

    try:
        _, val_loader = make_stl_loaders(
            data_dir=str(data_dir),
            img_size=96,
            batch_size=64,
            num_workers=0,
        )
    except Exception as exc:
        print(f"  [cortical_sheets] Could not create STL-10 loader: {exc}\n"
              "                    Skipping activation plots.")
        return

    for plot_idx, stem, phase_label in [
        (13, "stl_best",      "Post-STL Pretrain"),
        (14, "finetune_last", "Post-CIFAR Finetune"),
    ]:
        variant_sheets:   dict = {}
        layer_names_ord = None

        for label in VARIANT_LABELS:
            ckpt = ckpt_dir / f"{stem}_{label}.pt"
            if not ckpt.exists():
                continue
            try:
                model  = _load_model_for_viz(ckpt)
                sheets = _activation_cortical_sheets(model, val_loader)
            except Exception as exc:
                print(f"  [cortical_sheets] {label} ({stem}): {exc}")
                continue
            if sheets:
                variant_sheets[label] = sheets
                if layer_names_ord is None:
                    layer_names_ord = list(sheets.keys())

        if not variant_sheets or not layer_names_ord:
            print(f"  [cortical_sheets] No activation data for {phase_label}.")
            continue

        labels_ord = [l for l in VARIANT_LABELS if l in variant_sheets]
        _plot_sheet_grid(
            variant_sheets, layer_names_ord, labels_ord,
            title=f"Cortical Sheet — Mean Activation ({phase_label})",
            out_dir=out_dir,
            filename=f"{plot_idx}_cortical_sheets_activations_{stem}.png",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Selectivity diagrams
# ─────────────────────────────────────────────────────────────────────────────

def _collect_selectivity_sheets(
    model,
    layer_name: str,
    loader,
    n_classes: int,
    device: str = "cpu",
    n_samples_per_class: int = 100,
) -> Optional[np.ndarray]:
    """
    Register a forward hook on the named conv layer, run batches from *loader*,
    and accumulate per-class mean channel activations.

    Returns np.ndarray of shape (n_classes, H, W) — the selectivity map on the
    cortical sheet — or None if the layer is not found or no activations fire.

    selectivity[c, h, w] = mean_act_for_class_c[h, w]  −  global_mean[h, w]
    """
    import torch
    from topoloss.cortical_sheet.common import find_cortical_sheet_size

    # Locate layer by dotted name
    target = model
    try:
        for part in layer_name.split("."):
            target = getattr(target, part)
    except AttributeError:
        print(f"  [selectivity] Layer '{layer_name}' not found in model.")
        return None

    sums   = np.zeros(n_classes, dtype=np.float64)   # will be replaced
    n_ch   = None
    initialized = False
    counts = np.zeros(n_classes, dtype=np.int64)
    captured: list = []

    def _hook(mod, inp, out):
        # out: (B, C, H_f, W_f) — spatial-average over feature map
        captured.append(out.detach().cpu().float().mean(dim=(2, 3)))   # (B, C)

    handle = target.register_forward_hook(_hook)
    model.eval()

    try:
        with torch.no_grad():
            for imgs, labels in loader:
                if (counts >= n_samples_per_class).all():
                    break
                captured.clear()
                _ = model(imgs.to(device))
                if not captured:
                    continue
                acts = captured[0].numpy()   # (B, C)

                if not initialized:
                    n_ch  = acts.shape[1]
                    sums  = np.zeros((n_classes, n_ch), dtype=np.float64)
                    initialized = True

                for b in range(acts.shape[0]):
                    c = int(labels[b].item())
                    if 0 <= c < n_classes and counts[c] < n_samples_per_class:
                        sums[c]  += acts[b]
                        counts[c] += 1
    finally:
        handle.remove()

    if not initialized or n_ch is None:
        return None

    counts_safe = np.maximum(counts[:, None], 1)
    mean_act = (sums / counts_safe).astype(np.float32)   # (n_classes, C)

    global_mean = mean_act.mean(axis=0)                  # (C,)
    selectivity = mean_act - global_mean[None, :]        # (n_classes, C)

    size = find_cortical_sheet_size(n_ch)
    H, W = size.height, size.width
    # Reshape each class selectivity onto the (H, W) cortical sheet
    return selectivity[:, :H * W].reshape(n_classes, H, W)


def _plot_selectivity_grid(
    variant_maps: dict,
    class_names: list,
    labels_ordered: list,
    title: str,
    out_dir: Path,
    filename: str,
) -> None:
    """
    Plot a grid of selectivity heat-maps.
    Rows = stimulus categories.
    Columns = [category label] + model variants, each with its own colorbar.
    variant_maps: {label: np.ndarray(n_classes, H, W)}
    """
    n_classes  = len(class_names)
    n_variants = len(labels_ordered)
    n_cols     = 1 + n_variants   # label column + one per variant

    # Per-variant colour scale (independent dynamic range per column)
    col_vabs = {}
    for label in labels_ordered:
        maps = variant_maps.get(label)
        if maps is not None:
            col_vabs[label] = float(np.nanmax(np.abs(maps.reshape(-1)))) or 1.0
        else:
            col_vabs[label] = 1.0

    # Width ratios: label column narrow, then map column + small cbar gap per variant
    # We add extra horizontal space (wider per-variant allotment) for the colorbars
    width_ratios = [0.45] + [1.0] * n_variants

    fig, axes = plt.subplots(
        n_classes, n_cols,
        figsize=(0.45 * 3.2 + 4.2 * n_variants, 2.2 * n_classes),
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
    )
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(title, color="white", fontsize=12, y=1.005)

    cmap = plt.cm.RdGy_r.copy()
    cmap.set_bad("black")

    # ---- Column 0: category labels ----------------------------------------
    for row, cls in enumerate(class_names):
        ax = axes[row][0]
        ax.set_facecolor("#1a1a2e")
        ax.text(
            0.5, 0.5, cls,
            ha="center", va="center",
            color="white", fontsize=10, fontweight="bold",
            transform=ax.transAxes,
            wrap=False,
        )
        ax.axis("off")
        if row == 0:
            ax.set_title("Category", color="#aaaaaa", fontsize=9, fontweight="bold")

    # ---- Columns 1‥N: selectivity maps (per-variant colour scale) ----------
    col_images = {}   # store last imshow per column for colorbar
    for col_offset, label in enumerate(labels_ordered):
        col   = col_offset + 1
        maps  = variant_maps.get(label)   # (n_classes, H, W)
        vabs_c = col_vabs[label]
        for row in range(n_classes):
            ax = axes[row][col]
            if maps is None:
                ax.set_facecolor("#1a1a2e")
                ax.axis("off")
                continue
            im = ax.imshow(maps[row], cmap=cmap, vmin=-vabs_c, vmax=vabs_c,
                           aspect="auto", interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(DISPLAY_NAMES.get(label, label),
                             color="white", fontsize=9, fontweight="bold")
            col_images[col] = im

    # ---- Per-column colorbars (one per variant, spanning all rows) ----------
    for col_offset, label in enumerate(labels_ordered):
        col = col_offset + 1
        if col not in col_images:
            continue
        im_c   = col_images[col]
        vabs_c = col_vabs[label]
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=-vabs_c, vmax=vabs_c),
        )
        sm.set_array([])
        col_axes = list(axes[:, col])
        cbar = fig.colorbar(sm, ax=col_axes, location="right",
                            shrink=0.7, pad=0.04, aspect=25)
        cbar.ax.tick_params(colors="white", labelsize=6)
        cbar.set_label("selectivity", color="white", fontsize=7, labelpad=2)

    plt.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_selectivity_diagrams(
    data,
    ckpt_dir: Path,
    data_dir: Path,
    out_dir: Path,
    sel_layer: str = SEL_LAYER,
    n_samples_per_class: int = 100,
) -> None:
    """
    Plots 15–17 — Category selectivity maps on the cortical sheet.

    Three figures are produced:
      14  Post-STL-pretrain checkpoint  ×  STL-10 val stimuli
      15  Post-CIFAR-finetune checkpoint ×  CIFAR-10 val stimuli
      16  Post-CIFAR-finetune checkpoint ×  STL-10 val stimuli
           (shows how STL-10 selectivity is retained / forgotten after finetune)

    Each figure: rows = stimulus categories, cols = model variants.
    The selectivity map is computed at *sel_layer* (default: the final
    residual conv before average-pool, layer4.1.conv2, 512 channels → 16×32).
    """
    torch = _requires_torch()
    if torch is None:
        print("  [selectivity] PyTorch not available — skipping selectivity plots.")
        return

    _setup_model_paths()
    try:
        from stl_cifar_common import make_stl_loaders, make_cifar_loaders, make_cifar_overlap_loaders
    except ImportError as exc:
        print(f"  [selectivity] Cannot import stl_cifar_common: {exc}")
        return

    # Build loaders once (val only; small batch for memory efficiency)
    def _stl_loader():
        try:
            _, vl = make_stl_loaders(
                data_dir=str(data_dir), img_size=96,
                batch_size=64, num_workers=0,
            )
            return vl
        except Exception as exc:
            print(f"  [selectivity] STL-10 loader failed: {exc}")
            return None

    def _cifar_loader():
        try:
            _, vl = make_cifar_loaders(
                data_dir=str(data_dir), img_size=96,
                batch_size=64, num_workers=0,
            )
            return vl
        except Exception as exc:
            print(f"  [selectivity] CIFAR-10 loader failed: {exc}")
            return None

    def _cifar_overlap_loader():
        try:
            _, vl = make_cifar_overlap_loaders(
                data_dir=str(data_dir), img_size=96,
                batch_size=64, num_workers=0,
            )
            return vl
        except Exception as exc:
            print(f"  [selectivity] CIFAR-10 overlap loader failed: {exc}")
            return None

    scenarios = [
        # (plot_idx, ckpt_stem,       loader_fn,     class_names,    dataset_tag)
        (15, "stl_best",       _stl_loader,   STL10_CLASSES,  "STL-10"),
        (16, "finetune_last",  _cifar_overlap_loader, STL10_CLASSES,  "CIFAR-10 (overlap)"),
        (17, "finetune_last",  _stl_loader,   STL10_CLASSES,  "STL-10"),
    ]

    for plot_idx, ckpt_stem, loader_fn, class_names, dataset_tag in scenarios:
        loader = loader_fn()
        if loader is None:
            continue

        variant_maps: dict = {}
        for label in VARIANT_LABELS:
            ckpt = ckpt_dir / f"{ckpt_stem}_{label}.pt"
            if not ckpt.exists():
                print(f"  [selectivity] Checkpoint not found: {ckpt.name}")
                continue
            try:
                model = _load_model_for_viz(ckpt)
            except Exception as exc:
                print(f"  [selectivity] {label} ({ckpt_stem}): {exc}")
                continue

            sheet = _collect_selectivity_sheets(
                model, sel_layer, loader,
                n_classes=len(class_names),
                n_samples_per_class=n_samples_per_class,
            )
            if sheet is not None:
                variant_maps[label] = sheet

        if not variant_maps:
            print(f"  [selectivity] No data for plot {plot_idx}.")
            continue

        ckpt_phase = "Post-Pretrain" if ckpt_stem == "stl_best" else "Post-Finetune"
        labels_ord = [l for l in VARIANT_LABELS if l in variant_maps]
        _plot_selectivity_grid(
            variant_maps, class_names, labels_ord,
            title=(f"Category Selectivity — {ckpt_phase} × {dataset_tag}\n"
                   f"Layer: {sel_layer}"),
            out_dir=out_dir,
            filename=f"{plot_idx}_selectivity_{ckpt_stem}_{dataset_tag.lower().replace('-', '')}.png",
        )

        # ---- Per-category table ----
        col_w = 14
        header = f"Class".ljust(col_w)
        for lbl in labels_ord:
            header += DISPLAY_NAMES[lbl].rjust(col_w)
        print(f"\n─── Category Selectivity [mean |selectivity|] ─── {ckpt_phase} / {dataset_tag}")
        print(header)
        print("─" * (col_w + col_w * len(labels_ord)))
        for ci, cname in enumerate(class_names):
            row = cname.ljust(col_w)
            for lbl in labels_ord:
                sheet = variant_maps[lbl]          # (n_classes, H, W)
                val   = float(np.abs(sheet[ci]).mean()) if ci < sheet.shape[0] else float("nan")
                row  += f"{val:.4f}".rjust(col_w)
            print(row)
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Catastrophic forgetting analysis for STL→CIFAR experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Expected result files (in --results-dir):
              baseline_results_latest.json
              topo_only_results_latest.json
              topo_sparsity_results_latest.json
        """),
    )
    p.add_argument(
        "--results-dir", type=str,
        default=str(BASE_DIR / "outputs" / "stl_cifar" / "results"),
    )
    p.add_argument(
        "--out-dir", type=str,
        default=str(BASE_DIR / "outputs" / "stl_cifar" / "figures"),
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Exit 1 if any expected result file is missing",
    )
    p.add_argument(
        "--no-topo-plots", action="store_true",
        help="Skip topo/entropy loss plots",
    )
    p.add_argument(
        "--ckpt-dir", type=str,
        default=str(BASE_DIR / "outputs" / "stl_cifar" / "checkpoints"),
        help="Directory containing stl_best_*.pt / finetune_last_*.pt checkpoints",
    )
    p.add_argument(
        "--data-dir", type=str,
        default=str(BASE_DIR / "data" / "stl_cifar"),
        help="Root directory for STL-10 data (used for activation plots)",
    )
    p.add_argument(
        "--no-cortical-plots", action="store_true",
        help="Skip cortical sheet visualisation plots",
    )
    p.add_argument(
        "--no-selectivity-plots", action="store_true",
        help="Skip category selectivity diagram plots",
    )
    p.add_argument(
        "--sel-layer", type=str, default=SEL_LAYER,
        help="Named residual conv layer used for selectivity maps",
    )
    p.add_argument(
        "--sel-samples", type=int, default=100,
        help="Number of images per class to accumulate for selectivity maps",
    )
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    ckpt_dir    = Path(args.ckpt_dir)
    data_dir    = Path(args.data_dir)

    if not results_dir.exists():
        print(f"[ERROR] Results directory does not exist: {results_dir}")
        print("        Run the training scripts first.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nResults dir : {results_dir}")
    print(f"Output dir  : {out_dir}\n")

    print("Loading results …")
    data = load_results(results_dir, strict=args.strict)

    if not data:
        print("[ERROR] No result files could be loaded. Exiting.")
        sys.exit(1)

    print_table(data)
    save_text_summary(data, out_dir)

    print("Generating accuracy plots …")
    plot_stl_pretrain_acc(data, out_dir)
    plot_finetune_cifar_acc(data, out_dir)
    plot_finetune_stl_acc(data, out_dir)
    plot_forgetting_bar(data, out_dir)
    plot_accuracy_overview(data, out_dir)
    plot_stl_per_class_acc(data, out_dir)
    plot_stl_per_class_forgetting(data, out_dir)
    plot_finetuned_vs_unfinetuned_forgetting(data, out_dir)

    print("Generating CE loss plots …")
    plot_pretrain_ce_loss(data, out_dir)
    plot_finetune_ce_loss(data, out_dir)

    print("Generating gradient entropy plots …")
    plot_grad_entropy_curves(data, out_dir)
    print("Generating KL divergence plots …")
    plot_kl_curves(data, out_dir)
    plot_gradsurg_mask_density(data, out_dir)
    plot_gradsurg_cortical_entropy(data, out_dir)

    if not args.no_topo_plots:
        print("Generating topo/entropy loss plots …")
        plot_topo_loss_curves(data, out_dir)
        plot_entropy_loss_curves(data, out_dir)
        plot_l1sparse_loss_curves(data, out_dir)
        plot_gradsurg_finetune_comparison(data, out_dir)

    if not args.no_cortical_plots:
        print("Generating cortical sheet weight plots …")
        plot_cortical_sheets_weights(data, ckpt_dir, out_dir)
        print("Generating cortical sheet activation plots …")
        plot_cortical_sheets_activations(data, ckpt_dir, data_dir, out_dir)

    if not args.no_selectivity_plots:
        print("Generating category selectivity diagrams …")
        plot_selectivity_diagrams(
            data, ckpt_dir, data_dir, out_dir,
            sel_layer=args.sel_layer,
            n_samples_per_class=args.sel_samples,
        )

    print(f"\nAll done. Figures written to: {out_dir}\n")

if __name__ == "__main__":
    main()
