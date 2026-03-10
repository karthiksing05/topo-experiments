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

VARIANT_LABELS = ["baseline", "topo_only", "topo_sparsity", "topo_l1sparse"]
TOPO_VARIANTS  = ["topo_only", "topo_sparsity", "topo_l1sparse"]

COLORS = {
    "baseline":      "#7f8c8d",
    "topo_only":     "#2980b9",
    "topo_sparsity": "#27ae60",
    "topo_l1sparse": "#e67e22",
}
DISPLAY_NAMES = {
    "baseline":      "Baseline",
    "topo_only":     "Topo Only",
    "topo_sparsity": "Topo + Sparsity",
    "topo_l1sparse": "Topo + L1-Sparse",
}

STL10_CLASSES  = ["airplane", "bird",   "car",   "cat",   "deer",
                   "dog",     "horse",  "monkey", "ship",  "truck"]
CIFAR10_CLASSES = ["airplane", "automobile", "bird",  "cat",   "deer",
                   "dog",      "frog",       "horse", "ship",  "truck"]

# Default residual conv layer used for selectivity maps (ResNet-18 final block)
SEL_LAYER = "layer4.1.conv2"
MARKERS = {"baseline": "o", "topo_only": "s", "topo_sparsity": "^", "topo_l1sparse": "D"}


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
        title="Topographic Loss (LaplacianPyramid) — Topo Variants Only",
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
    """Plot 10b — KL divergence from uniform distribution across pretrain and finetune.

    Measures how far the batch-mean cortical activation distribution (softmax)
    deviates from uniform.  Large KL = some cortical regions dominate;
    small KL = activations spread evenly across the sheet.
    Active for topo_sparsity (lambda_kl > 0); zero for other variants.
    """
    _two_phase_loss_plot(
        data,
        pretrain_key="pretrain_kl_per_epoch",
        ft_keys=["ft_kl_per_epoch"],
        ylabel="KL Divergence from Uniform (batch-mean softmax)",
        title="Cortical KL Penalty — All Variants",
        out_path=out_dir / "10b_kl_curves.png",
        variants=VARIANT_LABELS,
    )


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
    Rows = stimulus categories, Columns = model variants.
    variant_maps: {label: np.ndarray(n_classes, H, W)}
    """
    n_classes  = len(class_names)
    n_variants = len(labels_ordered)

    fig, axes = plt.subplots(
        n_classes, n_variants,
        figsize=(3.2 * n_variants, 2.2 * n_classes),
        squeeze=False,
    )
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(title, color="white", fontsize=12, y=1.005)

    cmap = plt.cm.RdGy_r.copy()
    cmap.set_bad("black")

    # Shared colour scale across all panels
    all_vals = np.concatenate([
        v.reshape(-1) for v in variant_maps.values()
    ])
    vabs = float(np.nanmax(np.abs(all_vals))) or 1.0

    for col, label in enumerate(labels_ordered):
        maps = variant_maps.get(label)   # (n_classes, H, W)
        for row, cls in enumerate(class_names):
            ax = axes[row][col]
            if maps is None:
                ax.set_visible(False)
                continue
            im = ax.imshow(maps[row], cmap=cmap, vmin=-vabs, vmax=vabs,
                           aspect="auto", interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(DISPLAY_NAMES.get(label, label),
                             color="white", fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(cls, color="white", fontsize=8)
                ax.yaxis.set_label_position("left")

    # Shared colour bar
    cbar = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02)
    cbar.ax.tick_params(colors="white", labelsize=7)
    cbar.set_label("selectivity", color="white", fontsize=8)

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
            header += VARIANT_LABELS[lbl].rjust(col_w)
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

    print("Generating CE loss plots …")
    plot_pretrain_ce_loss(data, out_dir)
    plot_finetune_ce_loss(data, out_dir)

    print("Generating gradient entropy plots …")
    plot_grad_entropy_curves(data, out_dir)
    print("Generating KL divergence plots …")
    plot_kl_curves(data, out_dir)

    if not args.no_topo_plots:
        print("Generating topo/entropy loss plots …")
        plot_topo_loss_curves(data, out_dir)
        plot_entropy_loss_curves(data, out_dir)
        plot_l1sparse_loss_curves(data, out_dir)

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
