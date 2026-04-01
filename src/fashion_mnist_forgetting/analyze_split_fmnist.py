"""Analysis script for the Split-FashionMNIST continual learning experiment.

Reads outputs/split_fmnist/results/split_fmnist_results_latest.json
and writes all figures to outputs/split_fmnist/figures/.

Plots
-----
  1_acc_matrix_heatmaps.png    — 4-panel accuracy matrices (one per variant)
                                 rows = after training task j, cols = evaluated on task i
  2_bwt_comparison.png         — Backward Transfer bar chart (all variants)
  3_accuracy_overview.png      — 3-panel: learning acc / final acc / BWT side by side
  4_forgetting_per_task.png    — Per-task forgetting (acc[T-1][i] - acc[i][i]) grouped bars
  5_forgetting_trajectories.png — For tasks 0..T-2, accuracy degradation as later tasks train
  6_learning_curves.png        — Per-task training accuracy curves (diagonal of acc_matrix)
  7_ce_loss.png                — CE loss during each task's training (all variants)
  8_topo_loss.png              — TopoLoss during each task (topo variants only)
  9_grad_entropy.png           — Gradient entropy during each task (all variants)
  9b_act_entropy.png           — Activation entropy on cortical sheet during each task
  10_auxk_losses.png           — AuxK auxiliary loss (topo_auxk_pooled only)
  11_summary_table.png         — Text summary table: BWT / learning acc / final acc
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
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from fmnist_forgetting import SimpleNN
try:
    from topoloss.core import find_cortical_sheet_size
except ImportError:
    find_cortical_sheet_size = None

# ---------------------------------------------------------------------------
# Constants (must match split_fmnist_forgetting.py)
# ---------------------------------------------------------------------------

VARIANT_LABELS = [
    "baseline",
    "topo_only", "topo_sparsity", "topo_auxk", "topo_auxk_pooled",
    "topo_regionlock", "topo_regionlock_pooled",
    "ewc", "replay", "regionlock_notopo",
]
TOPO_VARIANTS  = ["topo_only", "topo_sparsity", "topo_auxk", "topo_auxk_pooled",
                  "topo_regionlock", "topo_regionlock_pooled"]

DISPLAY_NAMES = {
    "baseline":          "Baseline",
    "topo_only":         "Topo Only",
    "topo_sparsity":     "Topo + Sparsity",
    "topo_auxk":         "Topo + AuxK",
    "topo_auxk_pooled":  "Topo + AuxK (Pooled)",
    "topo_regionlock":        "Topo + RegionLock",
    "topo_regionlock_pooled": "Topo + RegionLock (Pooled)",
    "ewc":               "EWC",
    "replay":            "Replay Buffer",
    "regionlock_notopo": "RegionLock (no Topo)",
}

COLORS = {
    "baseline":          "#757575",
    "topo_only":         "#2196f3",
    "topo_sparsity":     "#4caf50",
    "topo_auxk":         "#ff5722",
    "topo_auxk_pooled":  "#9c27b0",
    "topo_regionlock":        "#e91e63",
    "topo_regionlock_pooled": "#3f51b5",
    "ewc":               "#00bcd4",
    "replay":            "#ff9800",
    "regionlock_notopo": "#795548",
}

MARKERS = {
    "baseline":          "o",
    "topo_only":         "s",
    "topo_sparsity":     "^",
    "topo_auxk":         "D",
    "topo_auxk_pooled":  "v",
    "topo_regionlock":        "P",
    "topo_regionlock_pooled": "<",
    "ewc":               "X",
    "replay":            "*",
    "regionlock_notopo": "h",
}

FMNIST_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",  "Shirt",   "Sneaker",  "Bag",   "AnkleBoot",
]

TASK_NAMES = [
    "T-shirt/Trouser",
    "Pullover/Dress",
    "Coat/Sandal",
    "Shirt/Sneaker",
    "Bag/AnkleBoot",
]

BASE_DIR    = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "outputs" / "split_fmnist" / "results"
FIGURES_DIR = BASE_DIR / "outputs" / "split_fmnist" / "figures"
CKPT_DIR    = BASE_DIR / "outputs" / "split_fmnist" / "checkpoints"

N_TASKS = 5

# Classes introduced at each task — must match split_fmnist_forgetting.py TASKS
TASK_CLASSES = [
    [0, 1],   # Task 0: T-shirt / Trouser
    [2, 3],   # Task 1: Pullover / Dress
    [4, 5],   # Task 2: Coat / Sandal
    [6, 7],   # Task 3: Shirt / Sneaker
    [8, 9],   # Task 4: Bag / AnkleBoot
]

VIS_LAYER_NAMES = ["fc1"]   # layer(s) to visualise


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


def _build_full_matrix(acc_matrix: list, n_tasks: int = N_TASKS) -> np.ndarray:
    """Convert ragged acc_matrix[j][i] (i≤j) to a full n×n array with NaN above diagonal."""
    mat = np.full((n_tasks, n_tasks), np.nan)
    for j, row in enumerate(acc_matrix):
        for i, val in enumerate(row):
            mat[j, i] = val
    return mat


def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Figure 1 — Accuracy matrix heatmaps (4 subplots, one per variant)
# ---------------------------------------------------------------------------

def fig_acc_matrix_heatmaps(results: dict, figures_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Accuracy Matrix — Split-FashionMNIST Continual Learning\n"
                 "Row = after training task $j$, Col = evaluated on task $i$",
                 fontsize=13, y=1.01)

    task_labels = [f"T{i}\n{n}" for i, n in enumerate(TASK_NAMES)]

    for ax, variant in zip(axes.flat, VARIANT_LABELS):
        r = results.get(variant)
        if r is None:
            ax.set_visible(False)
            continue

        mat = _build_full_matrix(r["acc_matrix"])
        im  = ax.imshow(mat, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")

        ax.set_xticks(range(N_TASKS))
        ax.set_yticks(range(N_TASKS))
        ax.set_xticklabels([f"T{i}" for i in range(N_TASKS)], fontsize=9)
        ax.set_yticklabels([f"After T{j}" for j in range(N_TASKS)], fontsize=9)
        ax.set_xlabel("Evaluated on task", fontsize=10)
        ax.set_ylabel("After training task", fontsize=10)
        ax.set_title(DISPLAY_NAMES[variant], fontsize=11,
                     color=COLORS[variant], fontweight="bold")

        # Annotate each cell with the accuracy value
        for j in range(N_TASKS):
            for i in range(j + 1):
                val = mat[j, i]
                if not np.isnan(val):
                    ax.text(i, j, f"{val:.0f}", ha="center", va="center",
                            fontsize=9, color="black" if 20 < val < 80 else "white",
                            fontweight="bold")

        # Mark the diagonal (just-learned) with a white border
        for i in range(N_TASKS):
            rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                  fill=False, edgecolor="white", linewidth=2.5)
            ax.add_patch(rect)

        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Accuracy (%)")

        # BWT annotation
        bwt = r.get("bwt", float("nan"))
        ax.text(0.02, 0.02, f"BWT = {bwt:+.1f}pp", transform=ax.transAxes,
                fontsize=9, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    _savefig(fig, figures_dir / "1_acc_matrix_heatmaps.png")


# ---------------------------------------------------------------------------
# Figure 2 — BWT comparison bar chart
# ---------------------------------------------------------------------------

def fig_bwt_comparison(results: dict, figures_dir: Path) -> None:
    variants   = [v for v in VARIANT_LABELS if v in results]
    bwt_vals   = [results[v].get("bwt", float("nan")) for v in variants]
    colors     = [COLORS[v] for v in variants]
    disp_names = [DISPLAY_NAMES[v] for v in variants]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(variants)), bwt_vals, color=colors, edgecolor="white",
                  linewidth=1.2, width=0.55)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(disp_names, rotation=15, ha="right", fontsize=11)
    ax.set_ylabel("Backward Transfer (pp)", fontsize=12)
    ax.set_title("Backward Transfer (BWT)\n"
                 "BWT < 0 = forgetting; BWT > 0 = positive backward transfer",
                 fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f"))

    for bar, val in zip(bars, bwt_vals):
        if not np.isnan(val):
            offset = 0.3 if val >= 0 else -0.3
            ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                    f"{val:+.1f}", ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=11, fontweight="bold")

    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _savefig(fig, figures_dir / "2_bwt_comparison.png")


# ---------------------------------------------------------------------------
# Figure 3 — Overview: learning acc / final acc / BWT
# ---------------------------------------------------------------------------

def fig_accuracy_overview(results: dict, figures_dir: Path) -> None:
    variants   = [v for v in VARIANT_LABELS if v in results]
    colors     = [COLORS[v] for v in variants]
    disp_names = [DISPLAY_NAMES[v] for v in variants]
    x          = np.arange(len(variants))
    width      = 0.28

    learn_accs = [results[v].get("learning_acc",   float("nan")) for v in variants]
    final_accs = [results[v].get("final_avg_acc",  float("nan")) for v in variants]
    bwt_vals   = [results[v].get("bwt",            float("nan")) for v in variants]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Split-FashionMNIST — Accuracy Overview", fontsize=14)

    for ax, vals, title, ylabel, fmt in [
        (axes[0], learn_accs, "Learning Accuracy\n(acc right after each task)",        "Avg Accuracy (%)",        ".1f"),
        (axes[1], final_accs, "Final Average Accuracy\n(all tasks, after all training)", "Avg Accuracy (%)",        ".1f"),
        (axes[2], bwt_vals,   "Backward Transfer (BWT)\n(negative = forgetting)",        "Backward Transfer (pp)",  "+.1f"),
    ]:
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=1.2, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(disp_names, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        if "BWT" in title:
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val,
                        f"{val:{fmt}}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    _savefig(fig, figures_dir / "3_accuracy_overview.png")


# ---------------------------------------------------------------------------
# Figure 4 — Per-task forgetting (acc[T-1][i] - acc[i][i]) grouped bars
# ---------------------------------------------------------------------------

def fig_forgetting_per_task(results: dict, figures_dir: Path) -> None:
    variants = [v for v in VARIANT_LABELS if v in results]
    n_tasks  = N_TASKS - 1   # task T-1 can't "forget" itself
    x        = np.arange(n_tasks)
    width    = 0.2

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("Per-Task Forgetting  (acc after last task − acc right after learning)",
                 fontsize=12)

    T = N_TASKS
    for idx, variant in enumerate(variants):
        r = results[variant]
        mat = _build_full_matrix(r["acc_matrix"])
        # Forgetting for task i = acc[T-1][i] - acc[i][i]
        forgetting = [mat[T - 1, i] - mat[i, i] for i in range(n_tasks)]
        offset = (idx - len(variants) / 2 + 0.5) * width
        ax.bar(x + offset, forgetting, width, label=DISPLAY_NAMES[variant],
               color=COLORS[variant], edgecolor="white", linewidth=0.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Task {i}\n{TASK_NAMES[i]}" for i in range(n_tasks)],
                       fontsize=10)
    ax.set_ylabel("Forgetting (pp)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f"))
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _savefig(fig, figures_dir / "4_forgetting_per_task.png")


# ---------------------------------------------------------------------------
# Figure 5 — Forgetting trajectories for early tasks over subsequent training
# ---------------------------------------------------------------------------

def fig_forgetting_trajectories(results: dict, figures_dir: Path) -> None:
    """For each earlier task i, plot accuracy on task i as tasks i+1..T-1 are trained."""
    T        = N_TASKS
    variants = [v for v in VARIANT_LABELS if v in results]
    # We only show tasks 0..T-2 as "earlier tasks" (T-1 can't be forgotten forward)
    n_rows   = min(T - 1, 4)   # at most 4 subplots

    fig, axes = plt.subplots(1, n_rows, figsize=(4 * n_rows, 4.5), sharey=False)
    if n_rows == 1:
        axes = [axes]
    fig.suptitle("Forgetting Trajectories — Accuracy on Earlier Tasks as Later Tasks Train",
                 fontsize=12)

    for plot_idx, earlier_task_i in enumerate(range(n_rows)):
        ax = axes[plot_idx]
        ax.set_title(f"Task {earlier_task_i}: {TASK_NAMES[earlier_task_i]}", fontsize=10)

        for variant in variants:
            r   = results[variant]
            mat = _build_full_matrix(r["acc_matrix"])

            # X: task index that was just trained (earlier_task_i .. T-1)
            # Y: accuracy on earlier_task_i right after that task finished
            x_vals = list(range(earlier_task_i, T))
            y_vals = [mat[j, earlier_task_i] for j in x_vals]

            ax.plot(x_vals, y_vals,
                    marker=MARKERS[variant], color=COLORS[variant],
                    label=DISPLAY_NAMES[variant], linewidth=1.8, markersize=6)

        ax.set_xlabel("Last task trained", fontsize=10)
        ax.set_ylabel("Task accuracy (%)", fontsize=10)
        ax.set_xticks(range(earlier_task_i, T))
        ax.set_xticklabels([f"T{j}" for j in range(earlier_task_i, T)], fontsize=9)
        ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)
        # Mark where task i itself was just trained (diagonal point)
        ax.axvline(earlier_task_i, color="gray", linewidth=1, linestyle=":")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(variants),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    _savefig(fig, figures_dir / "5_forgetting_trajectories.png")


# ---------------------------------------------------------------------------
# Figure 6 — Per-task learning curves (accuracy on current task while training)
# ---------------------------------------------------------------------------

def fig_learning_curves(results: dict, figures_dir: Path) -> None:
    """Accuracy on the CURRENT task during its training epochs."""
    variants = [v for v in VARIANT_LABELS if v in results]
    task_epochs = None
    for v in variants:
        r = results[v]
        if r["per_task_histories"]:
            task_epochs = len(r["per_task_histories"][0]["val_accs"])
            break
    if task_epochs is None:
        return

    fig, axes = plt.subplots(1, N_TASKS, figsize=(4 * N_TASKS, 4.5), sharey=True)
    fig.suptitle("Task Learning Curves — Accuracy on the Current Task During Training",
                 fontsize=12)

    for task_idx, ax in enumerate(axes):
        ax.set_title(f"Task {task_idx}\n{TASK_NAMES[task_idx]}", fontsize=10)
        for variant in variants:
            r    = results[variant]
            hist = r["per_task_histories"][task_idx]
            # val_accs[epoch] is a list of length task_idx+1; take index task_idx
            y = [ep_accs[task_idx] for ep_accs in hist["val_accs"]]
            ax.plot(range(1, len(y) + 1), y,
                    marker=MARKERS[variant], color=COLORS[variant],
                    label=DISPLAY_NAMES[variant], linewidth=1.8, markersize=5)

        ax.set_xlabel("Epoch", fontsize=10)
        if task_idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(variants),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    _savefig(fig, figures_dir / "6_learning_curves.png")


# ---------------------------------------------------------------------------
# Figure 7 — CE loss per task
# ---------------------------------------------------------------------------

def fig_ce_loss(results: dict, figures_dir: Path) -> None:
    variants = [v for v in VARIANT_LABELS if v in results]

    fig, axes = plt.subplots(1, N_TASKS, figsize=(4 * N_TASKS, 4), sharey=False)
    fig.suptitle("Cross-Entropy Loss During Each Task's Training", fontsize=12)

    for task_idx, ax in enumerate(axes):
        ax.set_title(f"Task {task_idx}: {TASK_NAMES[task_idx]}", fontsize=10)
        for variant in variants:
            r    = results[variant]
            hist = r["per_task_histories"][task_idx]
            y    = hist.get("ce", [])
            if y:
                ax.plot(range(1, len(y) + 1), y,
                        marker=MARKERS[variant], color=COLORS[variant],
                        label=DISPLAY_NAMES[variant], linewidth=1.8, markersize=5,
                        markevery=max(1, len(y) // 5))
        ax.set_xlabel("Epoch", fontsize=10)
        if task_idx == 0:
            ax.set_ylabel("CE Loss", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(variants),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    _savefig(fig, figures_dir / "7_ce_loss.png")


# ---------------------------------------------------------------------------
# Figure 8 — TopoLoss per task (topo variants only)
# ---------------------------------------------------------------------------

def fig_topo_loss(results: dict, figures_dir: Path) -> None:
    topo_present = [v for v in TOPO_VARIANTS if v in results]
    if not topo_present:
        return

    fig, axes = plt.subplots(1, N_TASKS, figsize=(4 * N_TASKS, 4), sharey=False)
    fig.suptitle("TopoLoss During Each Task's Training (topo variants)", fontsize=12)

    for task_idx, ax in enumerate(axes):
        ax.set_title(f"Task {task_idx}: {TASK_NAMES[task_idx]}", fontsize=10)
        for variant in topo_present:
            r    = results[variant]
            hist = r["per_task_histories"][task_idx]
            y    = hist.get("topo", [])
            if y and any(v > 0 for v in y):
                ax.plot(range(1, len(y) + 1), y,
                        marker=MARKERS[variant], color=COLORS[variant],
                        label=DISPLAY_NAMES[variant], linewidth=1.8, markersize=5,
                        markevery=max(1, len(y) // 5))
        ax.set_xlabel("Epoch", fontsize=10)
        if task_idx == 0:
            ax.set_ylabel("TopoLoss", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(topo_present),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    _savefig(fig, figures_dir / "8_topo_loss.png")


# ---------------------------------------------------------------------------
# Figure 9 — Gradient entropy per task
# ---------------------------------------------------------------------------

def fig_grad_entropy(results: dict, figures_dir: Path) -> None:
    variants = [v for v in VARIANT_LABELS if v in results]

    fig, axes = plt.subplots(1, N_TASKS, figsize=(4 * N_TASKS, 4), sharey=True)
    fig.suptitle("Gradient Entropy During Each Task's Training\n"
                 "(1 = uniform gradients; 0 = single-weight spike)",
                 fontsize=12)

    for task_idx, ax in enumerate(axes):
        ax.set_title(f"Task {task_idx}: {TASK_NAMES[task_idx]}", fontsize=10)
        for variant in variants:
            r    = results[variant]
            hist = r["per_task_histories"][task_idx]
            y    = hist.get("grad_entropy", [])
            if y:
                ax.plot(range(1, len(y) + 1), y,
                        marker=MARKERS[variant], color=COLORS[variant],
                        label=DISPLAY_NAMES[variant], linewidth=1.8, markersize=5,
                        markevery=max(1, len(y) // 5))
        ax.set_xlabel("Epoch", fontsize=10)
        if task_idx == 0:
            ax.set_ylabel("Gradient Entropy", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(variants),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    _savefig(fig, figures_dir / "9_grad_entropy.png")


# ---------------------------------------------------------------------------
# Figure 9b — Activation entropy per task
# ---------------------------------------------------------------------------

def fig_act_entropy(results: dict, figures_dir: Path) -> None:
    variants = [v for v in VARIANT_LABELS if v in results]

    fig, axes = plt.subplots(1, N_TASKS, figsize=(4 * N_TASKS, 4), sharey=True)
    fig.suptitle("Activation Entropy on Cortical Sheet During Each Task's Training\n"
                 "(1 = uniform / dense; 0 = single-unit spike / maximally sparse)",
                 fontsize=12)

    for task_idx, ax in enumerate(axes):
        ax.set_title(f"Task {task_idx}: {TASK_NAMES[task_idx]}", fontsize=10)
        for variant in variants:
            r    = results[variant]
            hist = r["per_task_histories"][task_idx]
            y    = hist.get("act_entropy", [])
            if y:
                ax.plot(range(1, len(y) + 1), y,
                        marker=MARKERS[variant], color=COLORS[variant],
                        label=DISPLAY_NAMES[variant], linewidth=1.8, markersize=5,
                        markevery=max(1, len(y) // 5))
        ax.set_xlabel("Epoch", fontsize=10)
        if task_idx == 0:
            ax.set_ylabel("Activation Entropy", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(variants),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    _savefig(fig, figures_dir / "9b_act_entropy.png")


# ---------------------------------------------------------------------------
# Figure 10 — AuxK losses (topo_auxk_pooled only)
# ---------------------------------------------------------------------------

def fig_auxk_losses(results: dict, figures_dir: Path) -> None:
    auxk_variants = [v for v in ["topo_auxk", "topo_auxk_pooled"] if v in results]
    if not auxk_variants:
        return

    fig, axes = plt.subplots(2, N_TASKS, figsize=(4 * N_TASKS, 7))
    fig.suptitle("AuxK Losses — topo_auxk and topo_auxk_pooled", fontsize=12)

    for task_idx in range(N_TASKS):
        ax0 = axes[0, task_idx]
        ax1 = axes[1, task_idx]

        for variant in auxk_variants:
            hist = results[variant]["per_task_histories"][task_idx]

            # Row 0: AuxK auxiliary CE loss
            y = hist.get("auxk_aux", [])
            if y:
                ax0.plot(range(1, len(y) + 1), y,
                         color=COLORS[variant], linewidth=1.8,
                         marker=MARKERS[variant], markersize=3,
                         label=DISPLAY_NAMES[variant],
                         markevery=max(1, len(y) // 5))

            # Row 1: Dead latent fraction
            y = hist.get("auxk_dead_frac", [])
            if y:
                ax1.plot(range(1, len(y) + 1), [v * 100 for v in y],
                         color=COLORS[variant], linewidth=1.8,
                         marker=MARKERS[variant], markersize=3,
                         label=DISPLAY_NAMES[variant],
                         markevery=max(1, len(y) // 5))

        ax0.set_title(f"Task {task_idx}: {TASK_NAMES[task_idx]}", fontsize=9)
        if task_idx == 0:
            ax0.set_ylabel("AuxK Aux Loss", fontsize=10)
            ax0.legend(fontsize=8, frameon=False)
        ax0.spines[["top", "right"]].set_visible(False)

        ax1.set_xlabel("Epoch", fontsize=10)
        if task_idx == 0:
            ax1.set_ylabel("Dead Latents (%)", fontsize=10)
        ax1.set_ylim(0, 105)
        ax1.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _savefig(fig, figures_dir / "10_auxk_losses.png")


# ---------------------------------------------------------------------------
# Figure 11 — Region similarity loss per task (topo_sparsity only)
# ---------------------------------------------------------------------------

def fig_sim_loss(results: dict, figures_dir: Path) -> None:
    if "topo_sparsity" not in results:
        return

    fig, axes = plt.subplots(1, N_TASKS, figsize=(4 * N_TASKS, 4), sharey=True)
    fig.suptitle(
        "Cortical Representational Similarity Penalty Per Task (topo_sparsity only)\n"
        "(mean sq off-diagonal cosine-sim between pooled regions; 0=orthogonal, 1=identical)",
        fontsize=11,
    )

    r = results["topo_sparsity"]
    for task_idx, ax in enumerate(axes):
        hist = r["per_task_histories"][task_idx]
        y = hist.get("sim", [])
        ax.set_title(f"Task {task_idx}: {TASK_NAMES[task_idx]}", fontsize=10)
        if y and any(v > 0 for v in y):
            ax.plot(
                range(1, len(y) + 1), y,
                marker=MARKERS["topo_sparsity"], color=COLORS["topo_sparsity"],
                label=DISPLAY_NAMES["topo_sparsity"], linewidth=1.8, markersize=5,
                markevery=max(1, len(y) // 5),
            )
        ax.set_xlabel("Epoch", fontsize=10)
        if task_idx == 0:
            ax.set_ylabel("Similarity Penalty", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _savefig(fig, figures_dir / "11_sim_loss.png")


# ---------------------------------------------------------------------------
# Figure 12 — Summary table
# ---------------------------------------------------------------------------

def fig_summary_table(results: dict, figures_dir: Path) -> None:
    T        = N_TASKS
    variants = [v for v in VARIANT_LABELS if v in results]

    # Build rows
    rows     = []
    col_hdrs = (
        ["Variant"]
        + [f"T{i}\nlearn" for i in range(T)]
        + [f"T{i}\nfinal" for i in range(T)]
        + ["BWT\n(pp)", "Final\nAvg (%)"]
    )

    for v in variants:
        r   = results[v]
        mat = _build_full_matrix(r["acc_matrix"])
        row = [DISPLAY_NAMES[v]]
        row += [f"{mat[i, i]:.1f}" for i in range(T)]
        row += [f"{mat[T - 1, i]:.1f}" for i in range(T)]
        row += [f"{r.get('bwt', float('nan')):+.1f}", f"{r.get('final_avg_acc', float('nan')):.1f}"]
        rows.append(row)

    fig, ax = plt.subplots(figsize=(max(14, len(col_hdrs) * 1.3), 2 + len(variants) * 0.6))
    ax.axis("off")

    t = ax.table(
        cellText=rows,
        colLabels=col_hdrs,
        cellLoc="center",
        loc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1.1, 1.5)

    # Colour header row
    for j in range(len(col_hdrs)):
        t[0, j].set_facecolor("#e0e0e0")
        t[0, j].set_text_props(fontweight="bold")

    # Colour variant name cells
    for i, v in enumerate(variants):
        t[i + 1, 0].set_facecolor(COLORS[v] + "40")  # 25% alpha
        t[i + 1, 0].set_text_props(fontweight="bold", color=COLORS[v])

    ax.set_title("Split-FashionMNIST — Summary Table\n"
                 "T_i learn = acc on task i right after training it; "
                 "T_i final = acc after ALL tasks",
                 fontsize=11, pad=20)
    plt.tight_layout()
    _savefig(fig, figures_dir / "11_summary_table.png")


# ---------------------------------------------------------------------------
# Model loading + activation helpers (for selectivity / activation figures)
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: Path, device: str, hidden_size: int = 256) -> "SimpleNN":
    """Load a SimpleNN (any sparsity_mode) from a .pt checkpoint."""
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    hs    = ckpt.get("hidden_size", hidden_size)
    mode  = ckpt.get("sparsity_mode", "topk_pooled" if "latent_counts" in state else "relu")
    fh    = ckpt.get("factor_h") or 4.0
    fw    = ckpt.get("factor_w") or 4.0

    # Derive safe k defaults for old checkpoints that didn't save k/k_aux
    if "k" in ckpt:
        k     = ckpt["k"]
        k_aux = ckpt.get("k_aux", k * 2)
    elif mode == "topk_pooled":
        size  = find_cortical_sheet_size(hs)
        pool_h = max(1, round(size.height / fh))
        pool_w = max(1, round(size.width  / fw))
        pool_n = pool_h * pool_w
        k      = min(1, pool_n)   # conservative: 1 active pooled unit
        k_aux  = min(4, pool_n)
    else:
        k, k_aux = 32, 64

    model = SimpleNN(
        hidden_size=hs,
        sparsity_mode=mode,
        k=k,
        k_aux=k_aux,
        factor_h=fh,
        factor_w=fw,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _make_val_loader(data_dir: str, batch_size: int = 512) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    ds = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


def _collect_acts(model: "SimpleNN", loader: DataLoader,
                  device: str, layer_names: list) -> tuple:
    """Run the full val set; return ({layer: Tensor(N,units)}, labels Tensor)."""
    acts_store = {n: [] for n in layer_names}
    lbls_store = []
    handles    = []
    for name in layer_names:
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
    return {n: torch.cat(v) for n, v in acts_store.items()}, torch.cat(lbls_store)


def _t_values(acts: torch.Tensor, labels: torch.Tensor, cls: int) -> torch.Tensor:
    """Per-unit Welch t-statistic: target class vs all others."""
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


def _act_to_sheet(act: torch.Tensor) -> np.ndarray:
    """Single activation vector → 2-D cortical sheet, min-max normalised."""
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


def _tv_to_sheet(tv: torch.Tensor) -> np.ndarray:
    """t-statistic vector → 2-D cortical sheet (un-normalised — caller sets vmin/vmax)."""
    if find_cortical_sheet_size is not None:
        size = find_cortical_sheet_size(tv.shape[0])
        arr  = tv[: size.height * size.width].reshape(size.height, size.width)
    else:
        side = int(tv.shape[0] ** 0.5)
        arr  = tv[: side * side].reshape(side, side)
    return arr.numpy()


# ---------------------------------------------------------------------------
# Figure 12 — Selectivity evolution across tasks (topo variants only)
# ---------------------------------------------------------------------------

def fig_selectivity_evolution(
    ckpt_dir: Path, data_dir: str, figures_dir: Path, device: str = "cpu"
) -> None:
    """For each topo variant: 10×5 grid showing fc1 t-statistic selectivity maps.

    Rows   = FashionMNIST classes (0-9)
    Cols   = task checkpoints (after tasks 0..4)
    Cells  = fc1 t-statistic cortical sheet for that class at that checkpoint.
             Greyed out when the class has not yet been trained on.
    Diagonal task band highlighted to show when each pair was introduced.

    Saved to: figures_dir/selectivity/12_selectivity_{variant}.png
    """
    if find_cortical_sheet_size is None:
        print("  [SKIP] selectivity — topoloss not importable")
        return

    sel_dir = figures_dir / "selectivity"
    sel_dir.mkdir(parents=True, exist_ok=True)

    loader = _make_val_loader(data_dir)

    # Colour band per task — which rows belong to which task
    task_row_colors = [
        "#e3f2fd",  # task 0 — light blue
        "#e8f5e9",  # task 1 — light green
        "#fff8e1",  # task 2 — light amber
        "#fce4ec",  # task 3 — light pink
        "#ede7f6",  # task 4 — light purple
    ]

    topo_present = [v for v in TOPO_VARIANTS if (ckpt_dir / f"last_task0_{v}.pt").exists()]
    if not topo_present:
        print("  [SKIP] selectivity — no topo checkpoints found")
        return

    for variant in topo_present:
        print(f"  Selectivity evolution for '{variant}' ...")

        # Load one checkpoint per task and compute full-val t-stats
        task_acts   = []   # list[dict[layer->Tensor]] per task checkpoint
        task_labels = []
        for task_idx in range(N_TASKS):
            ckpt_path = ckpt_dir / f"last_task{task_idx}_{variant}.pt"
            if not ckpt_path.exists():
                print(f"    [SKIP] missing {ckpt_path.name}")
                task_acts.append(None)
                task_labels.append(None)
                continue
            model = _load_model(ckpt_path, device)
            acts, labels = _collect_acts(model, loader, device, VIS_LAYER_NAMES)
            task_acts.append(acts)
            task_labels.append(labels)
            del model

        n_cls  = len(FMNIST_CLASSES)
        n_ckpt = N_TASKS
        fig, axes = plt.subplots(
            n_cls, n_ckpt,
            figsize=(n_ckpt * 2.5, n_cls * 2.5),
        )

        # Column headers
        for j in range(n_ckpt):
            axes[0, j].set_title(
                f"After Task {j}\n{TASK_NAMES[j]}", fontsize=8, pad=4,
                color="#333333",
            )

        # Which classes are visible after task j
        seen_after = [set(sum(TASK_CLASSES[:j+1], [])) for j in range(N_TASKS)]

        for row, cls in enumerate(range(n_cls)):
            # Which task introduced this class?
            task_of_cls = next(t for t, cl in enumerate(TASK_CLASSES) if cls in cl)
            row_bg = task_row_colors[task_of_cls]

            axes[row, 0].set_ylabel(
                f"{FMNIST_CLASSES[cls]}\n(T{task_of_cls})",
                fontsize=7, rotation=0, labelpad=52, va="center",
            )

            for col in range(n_ckpt):
                ax = axes[row, col]
                ax.set_facecolor(row_bg)

                # Highlight the task-introduction column for this class
                if col == task_of_cls:
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#ff6f00")
                        spine.set_linewidth(2)

                if cls not in seen_after[col] or task_acts[col] is None:
                    # Not yet trained on — show uniform grey
                    ax.imshow(
                        np.full((4, 4), 0.85), cmap="gray", vmin=0, vmax=1
                    )
                    ax.text(2, 2, "unseen", ha="center", va="center",
                            fontsize=6, color="#aaaaaa")
                    ax.axis("off")
                    continue

                tv   = _t_values(task_acts[col]["fc1"], task_labels[col], cls)
                tmap = _tv_to_sheet(tv)
                vmax = max(abs(tmap.min()), abs(tmap.max()), 0.1)
                im   = ax.imshow(tmap, cmap="RdGy_r", vmin=-vmax, vmax=vmax)
                ax.axis("off")

                # Add tiny colorbar only on the rightmost column
                if col == n_ckpt - 1:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                                 format="%.0f")

        fig.suptitle(
            f"fc1 Category Selectivity (t-statistic) — {DISPLAY_NAMES[variant]}\n"
            "Cols: model state after each task   |   "
            "Rows: all 10 FashionMNIST classes   |   "
            "Orange border = class first introduced   |   "
            "Grey = class not yet seen",
            fontsize=10, y=1.01,
        )
        plt.tight_layout()
        out_path = sel_dir / f"12_selectivity_{variant}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"    Saved -> {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 13 — Sample activation cortical sheets across tasks (topo variants)
# ---------------------------------------------------------------------------

def fig_sample_activations_evolution(
    ckpt_dir: Path, data_dir: str, figures_dir: Path, device: str = "cpu"
) -> None:
    """For each topo variant: 10×5 grid showing fc1 activation sheets.

    Rows   = FashionMNIST classes (0-9)
    Cols   = task checkpoints (after tasks 0..4)
    Cells  = fc1 cortical sheet for ONE representative validation image of that class
             at that checkpoint.  Greyed out when the class hasn't been trained yet.

    Saved to: figures_dir/activations/13_activations_{variant}.png
    """
    act_dir = figures_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    loader = _make_val_loader(data_dir)

    # Collect one example image per class from validation set
    example_imgs: dict = {}   # cls -> Tensor (1,1,28,28)
    for imgs, lbls in loader:
        for cls in range(len(FMNIST_CLASSES)):
            if cls not in example_imgs:
                idx = (lbls == cls).nonzero(as_tuple=True)[0]
                if len(idx):
                    example_imgs[cls] = imgs[idx[0]:idx[0] + 1]
        if len(example_imgs) == len(FMNIST_CLASSES):
            break

    def _single_layer_acts(model, img_t, layer_name):
        store   = {}
        handles = []
        layer   = getattr(model, layer_name)
        def _h(_m, _i, out):
            store[layer_name] = out[0].detach().cpu()
        handles.append(layer.register_forward_hook(_h))
        model.eval()
        with torch.no_grad():
            model(img_t.to(device))
        for h in handles:
            h.remove()
        return store.get(layer_name)

    task_row_colors = [
        "#e3f2fd", "#e8f5e9", "#fff8e1", "#fce4ec", "#ede7f6",
    ]

    topo_present = [v for v in TOPO_VARIANTS if (ckpt_dir / f"last_task0_{v}.pt").exists()]
    if not topo_present:
        print("  [SKIP] activations — no topo checkpoints found")
        return

    seen_after = [set(sum(TASK_CLASSES[:j+1], [])) for j in range(N_TASKS)]

    for variant in topo_present:
        print(f"  Sample activations evolution for '{variant}' ...")

        n_cls  = len(FMNIST_CLASSES)
        n_ckpt = N_TASKS
        # cols: input image | task-0 acts | task-1 acts | ... | task-4 acts
        n_cols = 1 + n_ckpt
        fig, axes = plt.subplots(
            n_cls, n_cols,
            figsize=(n_cols * 1.7, n_cls * 1.7),
        )

        # Headers
        axes[0, 0].set_title("Input", fontsize=8, pad=4)
        for j in range(n_ckpt):
            axes[0, j + 1].set_title(f"After T{j}", fontsize=8, pad=4)

        for row, cls in enumerate(range(n_cls)):
            task_of_cls = next(t for t, cl in enumerate(TASK_CLASSES) if cls in cl)
            row_bg = task_row_colors[task_of_cls]

            # Input image column
            img_t = example_imgs[cls]
            axes[row, 0].imshow(img_t[0, 0].numpy(), cmap="gray")
            axes[row, 0].set_ylabel(
                f"{FMNIST_CLASSES[cls]}\n(T{task_of_cls})",
                fontsize=7, rotation=0, labelpad=50, va="center",
            )
            axes[row, 0].axis("off")
            axes[row, 0].set_facecolor(row_bg)

            for col, task_idx in enumerate(range(n_ckpt)):
                ax = axes[row, col + 1]
                ax.set_facecolor(row_bg)

                if col == task_of_cls:
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#ff6f00")
                        spine.set_linewidth(2)

                if cls not in seen_after[task_idx]:
                    ax.imshow(np.full((4, 4), 0.85), cmap="gray", vmin=0, vmax=1)
                    ax.text(2, 2, "unseen", ha="center", va="center",
                            fontsize=6, color="#aaaaaa")
                    ax.axis("off")
                    continue

                ckpt_path = ckpt_dir / f"last_task{task_idx}_{variant}.pt"
                if not ckpt_path.exists():
                    ax.axis("off")
                    continue

                model = _load_model(ckpt_path, device)
                act   = _single_layer_acts(model, img_t, "fc1")
                del model
                if act is not None:
                    ax.imshow(_act_to_sheet(act), cmap="hot", vmin=0, vmax=1)
                ax.axis("off")

        fig.suptitle(
            f"fc1 Cortical Activation Sheets — {DISPLAY_NAMES[variant]}\n"
            "Cols: model state after each task   |   "
            "Rows: all 10 classes   |   "
            "Orange border = class first introduced   |   "
            "Grey = class not yet seen   |   min-max normalised",
            fontsize=10, y=1.01,
        )
        plt.tight_layout()
        out_path = act_dir / f"13_activations_{variant}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"    Saved -> {out_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze Split-FashionMNIST continual learning results"
    )
    p.add_argument(
        "--results",
        default=str(RESULTS_DIR / "split_fmnist_results_latest.json"),
        help="Path to results JSON (default: latest)",
    )
    p.add_argument(
        "--out-dir",
        default=str(FIGURES_DIR),
        help="Output directory for figures",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error if a required variant is missing from results",
    )
    p.add_argument(
        "--ckpt-dir",
        default=str(CKPT_DIR),
        help="Checkpoint directory for selectivity/activation figures "
             "(default: outputs/split_fmnist/checkpoints)",
    )
    p.add_argument(
        "--data-dir",
        default=str(BASE_DIR / "data"),
        help="FashionMNIST data directory (default: project-root/data)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device for model forward passes (default: cpu)",
    )
    args = p.parse_args()

    results_path = Path(args.results)
    figures_dir  = Path(args.out_dir)
    ckpt_dir     = Path(args.ckpt_dir)

    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    missing = [v for v in VARIANT_LABELS if v not in results]
    if missing:
        msg = f"Missing variants in results: {missing}"
        if args.strict:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}")

    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures in: {figures_dir}\n")

    # Present variants only
    available = {v: results[v] for v in VARIANT_LABELS if v in results}

    fig_acc_matrix_heatmaps(available,      figures_dir)
    fig_bwt_comparison(available,           figures_dir)
    fig_accuracy_overview(available,        figures_dir)
    fig_forgetting_per_task(available,      figures_dir)
    fig_forgetting_trajectories(available,  figures_dir)
    fig_learning_curves(available,          figures_dir)
    fig_ce_loss(available,                  figures_dir)
    fig_topo_loss(available,                figures_dir)
    fig_grad_entropy(available,             figures_dir)
    fig_act_entropy(available,              figures_dir)
    fig_auxk_losses(available,              figures_dir)
    fig_sim_loss(available,                 figures_dir)
    fig_summary_table(available,            figures_dir)

    print("\nGenerating selectivity evolution diagrams (topo variants) ...")
    fig_selectivity_evolution(ckpt_dir, args.data_dir, figures_dir, device=args.device)

    print("Generating sample activation evolution diagrams (topo variants) ...")
    fig_sample_activations_evolution(ckpt_dir, args.data_dir, figures_dir, device=args.device)

    print(f"\nAll figures written to: {figures_dir}")
    print("\nSummary:")
    print(f"  {'Variant':<22}  {'BWT':>8}   {'Learn Acc':>10}   {'Final Acc':>10}")
    print("  " + "-" * 58)
    for v in VARIANT_LABELS:
        if v not in results:
            continue
        r = results[v]
        print(
            f"  {DISPLAY_NAMES[v]:<22}  {r.get('bwt', float('nan')):>+7.2f}pp"
            f"   {r.get('learning_acc', float('nan')):>9.1f}%"
            f"   {r.get('final_avg_acc', float('nan')):>9.1f}%"
        )


if __name__ == "__main__":
    main()
