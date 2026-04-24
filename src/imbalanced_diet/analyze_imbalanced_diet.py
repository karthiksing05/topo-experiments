"""analyze_imbalanced_diet.py

Loads the JSON results produced by imbalanced_diet_topo.py and generates
six publication-quality figures:

  Figure 1 — Cortical Selectivity Atlas
      Grid (variants × diets): each cell shows the 16×16 cortical sheet
      coloured by the class each neuron is most selective for.  Multiple
      class colours appear on the same sheet.  Brightness encodes the
      strength of selectivity (t-statistic magnitude).

  Figure 2 — Class Territory Under Imbalanced Training
      Stacked-bar chart: fraction of the cortical sheet "owned" by each
      class, grouped by training diet and variant.  Reveals how minority
      classes lose cortical territory.

  Figure 3 — Spatial Clustering Score (SCS) Comparison
      Grouped bar chart: SCS (fraction of 4-connected neighbours sharing
      the same dominant class) per diet × variant.  A higher value means
      same-class neurons cluster together spatially.

  Figure 4 — Per-Class Validation Accuracy Heatmaps
      One subplot per class: accuracy as a colour grid (diets × variants).
      Shows which variant best preserves minority-class representations.

  Figure 5 — Training Loss Curves
      CE loss over epochs for selected diets (balanced + extreme_skew),
      all four variants overlaid.

  Figure 6 — Per-Class Activation Heatmaps on Cortical Sheet
      Mean activation heatmap coloured per-class to show which cortical
      regions "light up" for each category under balanced vs extreme_skew.

All figures are saved to:
  outputs/imbalanced_diet/figures/

Usage
-----
python src/imbalanced_diet/analyze_imbalanced_diet.py
python src/imbalanced_diet/analyze_imbalanced_diet.py \\
    --results outputs/imbalanced_diet/results/imbalanced_diet_results_latest.json \\
    --output-dir outputs/imbalanced_diet/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

try:
    from scipy.ndimage import gaussian_filter as _gaussian_filter
    def gaussian_filter(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return _gaussian_filter(arr, sigma=sigma)
except ImportError:
    def gaussian_filter(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:  # type: ignore[misc]
        """Trivial fallback when scipy is not installed."""
        return arr

# Resolve project root (this file is two directories below the root)
BASE_DIR   = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs" / "imbalanced_diet"

# ── Shared constants (mirrored from training script) ────────────────────────

ORIG_CLASS_INDICES = [0, 1, 7, 8]
CLASS_NAMES   = ["T-shirt", "Trouser", "Sneaker", "Bag"]
N_CLASSES     = 4

CLASS_HEX = {
    "T-shirt": "#D53E4F",
    "Trouser": "#3264C1",
    "Sneaker": "#2CA02C",
    "Bag":     "#E67300",
}
CLASS_RGB_FLOAT = {
    "T-shirt": np.array([213/255,  62/255,  79/255]),
    "Trouser": np.array([ 50/255, 100/255, 193/255]),
    "Sneaker": np.array([ 44/255, 160/255,  44/255]),
    "Bag":     np.array([230/255, 115/255,   0/255]),
}

DIET_ORDER = [
    "balanced", "top_dominant", "bottom_dominant",
    "footwear_dominant", "bag_dominant", "extreme_skew",
]
DIET_SHORT = {
    "balanced":          "Balanced\n25/25/25/25",
    "top_dominant":      "Top Dom.\n70/10/10/10",
    "bottom_dominant":   "Bottom Dom.\n10/70/10/10",
    "footwear_dominant": "Footwear Dom.\n10/10/70/10",
    "bag_dominant":      "Bag Dom.\n10/10/10/70",
    "extreme_skew":      "Extreme\n55/25/15/5",
}

VARIANT_ORDER  = ["baseline", "topo_r2", "topo_r4", "topo_r8"]
VARIANT_LABELS = {
    "baseline": "Baseline",
    "topo_r2":  "Topo τ=5 r=2",
    "topo_r4":  "Topo τ=5 r=4",
    "topo_r8":  "Topo τ=5 r=8",
}
VARIANT_COLORS = {
    "baseline": "#757575",
    "topo_r2":  "#2196F3",
    "topo_r4":  "#4CAF50",
    "topo_r8":  "#FF5722",
}

# ── Utilities ─────────────────────────────────────────────────────────────────

def load_results(path: str | Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _get(results: dict, diet: str, variant: str) -> dict | None:
    return results.get(f"{diet}__{variant}")


def _cortical_dims(hidden_size: int) -> tuple[int, int]:
    """Return (H, W) of the cortical sheet for *hidden_size* neurons.

    Uses the same topoloss find_cortical_sheet_size logic.  For 256: (16, 16).
    We reproduce the calculation without importing topoloss (pure-Python) so the
    analysis script can run without the GPU environment.
    """
    # Find closest rectangular factorisation: height <= width, h*w = hidden_size
    h = int(hidden_size ** 0.5)
    while h > 1 and hidden_size % h != 0:
        h -= 1
    return h, hidden_size // h


def t_stats_to_rgb(
    t_stats: np.ndarray,
    hidden_size: int,
    gamma: float = 0.6,
) -> np.ndarray:
    """Convert (n_neurons, N_CLASSES) t-statistics to an RGB cortical-sheet image.

    Winner-takes-all colouring: each neuron is assigned the colour of its
    most-selective class.  Pixel brightness is proportional to the t-stat
    of the winning class (gamma-corrected for visibility).

    Neurons whose maximum positive t-stat is below 5% of the global maximum
    are rendered as light grey (unselective / weakly-driven).

    Returns
    -------
    rgb : ndarray  shape (H, W, 3), dtype float32, values in [0, 1].
    """
    H, W = _cortical_dims(hidden_size)
    n    = H * W

    t_pos  = np.maximum(t_stats[:n], 0.0)           # (n, N_CLASSES)
    t_max  = t_pos.max(axis=1)                       # (n,)   per-neuron max
    t_dom  = t_pos.argmax(axis=1)                    # (n,)   dominant class
    global_max = t_max.max() if t_max.max() > 0 else 1.0
    strength   = (t_max / global_max) ** gamma       # (n,)   brightness [0,1]

    GREY = np.array([0.88, 0.88, 0.88], dtype=np.float32)
    rgb  = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        if strength[i] < 0.05:
            rgb[i] = GREY
        else:
            c_name   = CLASS_NAMES[t_dom[i]]
            cls_col  = CLASS_RGB_FLOAT[c_name].astype(np.float32)
            # Blend class colour toward white with decreasing strength
            rgb[i]   = cls_col * strength[i] + np.ones(3, dtype=np.float32) * (1.0 - strength[i])
            rgb[i]   = np.clip(rgb[i], 0.0, 1.0)

    return rgb.reshape(H, W, 3)


def t_stats_to_blended_rgb(
    t_stats: np.ndarray,
    hidden_size: int,
) -> np.ndarray:
    """Soft blend: each pixel is the weighted average of class colours by t-stat.

    Returns RGB array shape (H, W, 3).
    """
    H, W = _cortical_dims(hidden_size)
    n    = H * W

    t_pos = np.maximum(t_stats[:n], 0.0)             # (n, N_CLASSES)
    total = t_pos.sum(axis=1, keepdims=True)          # (n, 1)
    GREY  = np.array([0.88, 0.88, 0.88], dtype=np.float32)

    # Stack class colours into (N_CLASSES, 3)
    cols = np.stack([CLASS_RGB_FLOAT[cn] for cn in CLASS_NAMES], axis=0).astype(np.float32)

    rgb  = np.zeros((n, 3), dtype=np.float32)
    unsel = total[:, 0] < 1e-3                        # unselective mask

    weights         = t_pos / total.clip(min=1e-10)   # (n, N_CLASSES)
    rgb[:]          = weights @ cols                   # (n, 3)
    rgb[unsel]      = GREY
    return rgb.reshape(H, W, 3)


def activation_to_rgb(
    heatmap: np.ndarray,
    hidden_size: int,
    cmap: str = "inferno",
) -> np.ndarray:
    """Single activation heatmap → RGB using a matplotlib colourmap."""
    H, W = _cortical_dims(hidden_size)
    cmap_fn = plt.get_cmap(cmap)
    flat = np.array(heatmap[:H * W])
    flat = (flat - flat.min()) / (flat.ptp() + 1e-10)
    return cmap_fn(flat)[:, :3].reshape(H, W, 3).astype(np.float32)


# ── Figure helpers ────────────────────────────────────────────────────────────

def _class_legend_patches() -> list:
    return [
        mpatches.Patch(color=CLASS_HEX[name], label=name)
        for name in CLASS_NAMES
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: Cortical Selectivity Atlas
# ═════════════════════════════════════════════════════════════════════════════

def plot_cortical_atlas(
    results: dict,
    hidden_size: int = 256,
    out_dir: Path = OUTPUT_DIR / "figures",
    show_blended: bool = True,
) -> None:
    """Winner-takes-all cortical selectivity maps, variants × diets grid."""
    n_v = len(VARIANT_ORDER)
    n_d = len(DIET_ORDER)

    # Compute number of panel columns: WTA map + optionally blended map
    n_col_per_diet = 2 if show_blended else 1
    total_cols = n_d * n_col_per_diet

    fig_w = max(20, total_cols * 1.6)
    fig_h = max(10, n_v * 2.0 + 1.5)
    fig, axes = plt.subplots(
        n_v, total_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )

    for vi, variant in enumerate(VARIANT_ORDER):
        for di, diet in enumerate(DIET_ORDER):
            r = _get(results, diet, variant)

            # WTA panel
            col_wta = di * n_col_per_diet
            ax = axes[vi, col_wta]
            if r is not None:
                t   = np.array(r["t_stats"], dtype=np.float32)
                rgb = t_stats_to_rgb(t, hidden_size)
                ax.imshow(rgb, interpolation="nearest", aspect="equal")
            else:
                ax.set_facecolor("#cccccc")
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="red")

            ax.set_xticks([]);  ax.set_yticks([])
            if di == 0:
                ax.set_ylabel(VARIANT_LABELS[variant], fontsize=8, labelpad=4)
            if vi == 0:
                ax.set_title(DIET_SHORT[diet].split("\n")[0], fontsize=8)

            # Blended panel
            if show_blended and r is not None:
                col_bl = di * n_col_per_diet + 1
                ax2 = axes[vi, col_bl]
                t    = np.array(r["t_stats"], dtype=np.float32)
                rgb2 = t_stats_to_blended_rgb(t, hidden_size)
                ax2.imshow(rgb2, interpolation="nearest", aspect="equal")
                ax2.set_xticks([]);  ax2.set_yticks([])
                # Thin divider border
                for spine in ax2.spines.values():
                    spine.set_edgecolor("#aaaaaa")
                    spine.set_linewidth(0.5)

    # Diet labels spanning both sub-panels
    for di, diet in enumerate(DIET_ORDER):
        col_center = di * n_col_per_diet + (n_col_per_diet - 1) / 2
        if vi == 0:
            mid_ax = axes[0, di * n_col_per_diet]
            mid_ax.set_title(DIET_SHORT[diet], fontsize=7.5)

    # Column sub-headers
    if show_blended:
        for di in range(n_d):
            axes[0, di * 2    ].set_title(
                DIET_SHORT[DIET_ORDER[di]] + "\n(WTA)", fontsize=6.5)
            axes[0, di * 2 + 1].set_title(
                DIET_SHORT[DIET_ORDER[di]] + "\n(blend)", fontsize=6.5)

    # Legend
    handles = _class_legend_patches()
    fig.legend(handles=handles, loc="lower center", ncol=N_CLASSES,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Cortical Selectivity Atlas — Neurons Coloured by Dominant Class\n"
        "(Brightness = selectivity strength  |  WTA = winner-takes-all  |  blend = class-weighted)",
        fontsize=11, y=1.01,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = out_dir / "fig1_cortical_atlas.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: Class Territory Stacked Bars
# ═════════════════════════════════════════════════════════════════════════════

def plot_class_territories(
    results: dict,
    out_dir: Path = OUTPUT_DIR / "figures",
) -> None:
    """Fraction of cortical sheet owned by each class, per (diet, variant)."""
    n_v = len(VARIANT_ORDER)
    n_d = len(DIET_ORDER)
    bar_width = 0.18
    group_gap = 0.15
    group_width = n_v * bar_width + group_gap

    fig, ax = plt.subplots(figsize=(16, 5))
    x_centers = np.arange(n_d) * group_width

    for vi, variant in enumerate(VARIANT_ORDER):
        bottoms = np.zeros(n_d)
        x_pos   = x_centers + vi * bar_width - (n_v - 1) * bar_width / 2

        for ci in range(N_CLASSES):
            heights = []
            for di, diet in enumerate(DIET_ORDER):
                r = _get(results, diet, variant)
                if r is not None:
                    heights.append(r["class_territories"][ci] * 100.0)
                else:
                    heights.append(0.0)
            heights = np.array(heights)

            bars = ax.bar(
                x_pos, heights, bar_width,
                bottom=bottoms,
                color=CLASS_HEX[CLASS_NAMES[ci]],
                alpha=0.82 if ci == 0 else (0.75 - ci * 0.05),
                label=CLASS_NAMES[ci] if vi == 0 else "_",
                edgecolor="white", linewidth=0.3,
            )
            bottoms += heights

        # Variant label below each group of bars
        for di in range(n_d):
            ax.text(
                x_pos[di], -5, VARIANT_LABELS[variant],
                ha="center", va="top", fontsize=5.5, color=VARIANT_COLORS[variant],
                rotation=45,
            )

    # Training diet annotation lines (ground-truth proportions)
    for di, diet in enumerate(DIET_ORDER):
        proportions = {
            "balanced":          [25, 25, 25, 25],
            "top_dominant":      [70, 10, 10, 10],
            "bottom_dominant":   [10, 70, 10, 10],
            "footwear_dominant": [10, 10, 70, 10],
            "bag_dominant":      [10, 10, 10, 70],
            "extreme_skew":      [55, 25, 15,  5],
        }[diet]
        cum = 0
        for ci, p in enumerate(proportions):
            ax.plot(
                [x_centers[di] - group_width * 0.45,
                 x_centers[di] + group_width * 0.45],
                [cum + p, cum + p],
                color=CLASS_HEX[CLASS_NAMES[ci]],
                linewidth=1.5, linestyle="--", alpha=0.55, zorder=5,
            )
            cum += p

    ax.set_xlim(x_centers[0] - group_width, x_centers[-1] + group_width)
    ax.set_ylim(-12, 108)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([DIET_SHORT[d].replace("\n", "\n") for d in DIET_ORDER],
                       fontsize=8)
    ax.set_ylabel("% of cortical sheet", fontsize=10)
    ax.set_xlabel("Training diet  (dashed lines = ground-truth training proportions)", fontsize=9)
    ax.set_title(
        "Class Cortical Territory Under Imbalanced Training Diets\n"
        "(each colour = fraction of neurons selective for that class)",
        fontsize=11,
    )

    handles = _class_legend_patches()
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    # Variant legend in bottom-right
    variant_handles = [
        mpatches.Patch(color=VARIANT_COLORS[v], label=VARIANT_LABELS[v])
        for v in VARIANT_ORDER
    ]
    ax.legend(handles=handles + variant_handles, loc="upper right",
              fontsize=7.5, framealpha=0.9, ncol=2)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig2_class_territories.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3: Spatial Clustering Score
# ═════════════════════════════════════════════════════════════════════════════

def plot_spatial_clustering(
    results: dict,
    out_dir: Path = OUTPUT_DIR / "figures",
) -> None:
    """Grouped bars: spatial contiguity score per (diet × variant)."""
    n_v = len(VARIANT_ORDER)
    n_d = len(DIET_ORDER)
    bar_width = 0.18
    group_gap = 0.10

    fig, ax = plt.subplots(figsize=(13, 4.5))
    x_centers = np.arange(n_d) * (n_v * bar_width + group_gap)

    # Baseline expected SCS under random assignment for each diet
    chance = {
        "balanced":          0.25,
        "top_dominant":      0.70**2 + 3 * 0.10**2,
        "bottom_dominant":   0.70**2 + 3 * 0.10**2,
        "footwear_dominant": 0.70**2 + 3 * 0.10**2,
        "bag_dominant":      0.70**2 + 3 * 0.10**2,
        "extreme_skew":      0.55**2 + 0.25**2 + 0.15**2 + 0.05**2,
    }

    for vi, variant in enumerate(VARIANT_ORDER):
        scs_vals = []
        for di, diet in enumerate(DIET_ORDER):
            r = _get(results, diet, variant)
            scs_vals.append(r["spatial_clustering_score"] if r else 0.0)

        x_pos = x_centers + (vi - (n_v - 1) / 2) * bar_width
        ax.bar(x_pos, scs_vals, bar_width,
               color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant],
               alpha=0.82, edgecolor="white", linewidth=0.5)

    # Chance level dashed lines per diet group
    for di, diet in enumerate(DIET_ORDER):
        ax.hlines(
            chance[diet],
            x_centers[di] - n_v * bar_width * 0.6,
            x_centers[di] + n_v * bar_width * 0.6,
            colors="#333333", linestyles=":", linewidth=1.2,
            label="Chance" if di == 0 else "_",
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels([DIET_SHORT[d] for d in DIET_ORDER], fontsize=8)
    ax.set_ylabel("Spatial Contiguity Score (SCS)", fontsize=10)
    ax.set_xlabel("Training diet", fontsize=9)
    ax.set_title(
        "Topographic Clustering: Fraction of Neighbours Sharing the Same Dominant Class\n"
        "(dashed line = chance level under random assignment)",
        fontsize=11,
    )
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, ncol=3, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig3_spatial_clustering.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4: Per-Class Accuracy Heatmaps
# ═════════════════════════════════════════════════════════════════════════════

def plot_per_class_accuracy_heatmaps(
    results: dict,
    out_dir: Path = OUTPUT_DIR / "figures",
) -> None:
    """Accuracy grid: diets × variants, one subplot per class."""
    n_v = len(VARIANT_ORDER)
    n_d = len(DIET_ORDER)

    fig, axes = plt.subplots(1, N_CLASSES, figsize=(16, 4),
                             gridspec_kw={"wspace": 0.35})

    for ci, cls_name in enumerate(CLASS_NAMES):
        ax = axes[ci]
        grid = np.full((n_d, n_v), np.nan)

        for di, diet in enumerate(DIET_ORDER):
            for vi, variant in enumerate(VARIANT_ORDER):
                r = _get(results, diet, variant)
                if r is not None:
                    grid[di, vi] = r["final_val_acc_per_class"][ci]

        im = ax.imshow(grid, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(n_v))
        ax.set_xticklabels(
            [VARIANT_LABELS[v].replace(" ", "\n") for v in VARIANT_ORDER],
            fontsize=6.5,
        )
        ax.set_yticks(range(n_d))
        ax.set_yticklabels([DIET_SHORT[d] for d in DIET_ORDER], fontsize=6.5)
        ax.set_title(f"{cls_name}\n(class {ci})", fontsize=9,
                     color=CLASS_HEX[cls_name], fontweight="bold")

        # Annotate cells
        for di in range(n_d):
            for vi in range(n_v):
                val = grid[di, vi]
                if not np.isnan(val):
                    ax.text(vi, di, f"{val:.0f}",
                            ha="center", va="center", fontsize=6,
                            color="black" if 30 < val < 80 else "white")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Accuracy (%)")

    fig.suptitle(
        "Per-Class Validation Accuracy (%) Under Imbalanced Training Diets",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig4_perclass_accuracy.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 5: Training Loss Curves
# ═════════════════════════════════════════════════════════════════════════════

def plot_training_curves(
    results: dict,
    selected_diets: list[str] | None = None,
    out_dir: Path = OUTPUT_DIR / "figures",
) -> None:
    """CE loss vs epoch for all variants, shown for selected diets."""
    if selected_diets is None:
        selected_diets = ["balanced", "extreme_skew", "top_dominant", "bag_dominant"]

    n_d = len(selected_diets)
    fig, axes = plt.subplots(1, n_d, figsize=(5 * n_d, 4),
                             gridspec_kw={"wspace": 0.28})
    if n_d == 1:
        axes = [axes]

    for di, diet in enumerate(selected_diets):
        ax = axes[di]
        for variant in VARIANT_ORDER:
            r = _get(results, diet, variant)
            if r is None:
                continue
            ce_hist = r["history"]["ce"]
            ax.plot(
                range(1, len(ce_hist) + 1),
                ce_hist,
                color=VARIANT_COLORS[variant],
                label=VARIANT_LABELS[variant],
                linewidth=1.8,
                alpha=0.85,
            )

        ax.set_title(DIET_SHORT[diet], fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("CE loss" if di == 0 else "", fontsize=8)
        ax.legend(fontsize=6.5, framealpha=0.8)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Cross-Entropy Training Loss Curves Under Imbalanced Training Diets",
        fontsize=11,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig5_training_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 6: Per-Class Activation Heatmaps on Cortical Sheet
# ═════════════════════════════════════════════════════════════════════════════

def plot_activation_heatmaps(
    results: dict,
    hidden_size: int = 256,
    selected_diets: list[str] | None = None,
    out_dir: Path = OUTPUT_DIR / "figures",
) -> None:
    """For each (selected diet, variant) pair, show per-class mean activation."""
    if selected_diets is None:
        selected_diets = ["balanced", "extreme_skew"]

    H, W = _cortical_dims(hidden_size)
    n_d  = len(selected_diets)
    n_v  = len(VARIANT_ORDER)

    # Rows = (diet × variant) pairs; Cols = N_CLASSES individual heatmaps + 1 combined
    n_rows = n_d * n_v
    n_cols = N_CLASSES + 1   # one per class + one combined

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.8, n_rows * 1.8),
        gridspec_kw={"hspace": 0.08, "wspace": 0.05},
    )

    row = 0
    for diet in selected_diets:
        for variant in VARIANT_ORDER:
            r = _get(results, diet, variant)

            for ci, cls_name in enumerate(CLASS_NAMES):
                ax = axes[row, ci]
                if r is not None:
                    cls_hm = np.array(r["class_activation_heatmaps"][ci],
                                      dtype=np.float32)   # (n_neurons,)
                    mx = cls_hm.max()
                    if mx > 0:
                        cls_hm = cls_hm / mx
                    cls_hm_2d = gaussian_filter(cls_hm[:H * W].reshape(H, W),
                                                sigma=0.8)

                    # Custom single-class colourmap: white → class_colour
                    base_col = np.array(
                        [int(CLASS_HEX[cls_name][i:i+2], 16) / 255
                         for i in (1, 3, 5)]
                    )
                    cmap_arr = np.zeros((256, 4))
                    for k in range(256):
                        t = k / 255.0
                        cmap_arr[k, :3] = (1.0 - t) * np.ones(3) + t * base_col
                        cmap_arr[k,  3] = 1.0
                    cmap_custom = matplotlib.colors.ListedColormap(cmap_arr)
                    ax.imshow(cls_hm_2d, cmap=cmap_custom, vmin=0, vmax=1,
                              interpolation="bilinear", aspect="equal")
                else:
                    ax.set_facecolor("#cccccc")

                ax.set_xticks([]); ax.set_yticks([])
                if row % n_v == 0 and row // n_v == 0:
                    ax.set_title(cls_name, fontsize=8,
                                 color=CLASS_HEX[cls_name], fontweight="bold")
                if ci == 0:
                    ax.set_ylabel(
                        f"{VARIANT_LABELS[variant]}\n{DIET_SHORT[diet]}",
                        fontsize=6.0, labelpad=2,
                    )

            # Combined "domination" map (last column)
            ax_c = axes[row, N_CLASSES]
            if r is not None:
                t   = np.array(r["t_stats"], dtype=np.float32)
                rgb = t_stats_to_rgb(t, hidden_size)
                ax_c.imshow(rgb, interpolation="nearest", aspect="equal")
                ax_c.set_title("Selectivity\n(WTA)", fontsize=7.5) \
                    if row == 0 else None
            ax_c.set_xticks([]); ax_c.set_yticks([])

            row += 1

    # Column headers
    for ci, cls_name in enumerate(CLASS_NAMES):
        axes[0, ci].set_title(
            f"Mean Act.\n{cls_name}", fontsize=7.5,
            color=CLASS_HEX[cls_name], fontweight="bold",
        )

    handles = _class_legend_patches()
    fig.legend(handles=handles, loc="lower center", ncol=N_CLASSES,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Per-Class Mean Activation Heatmaps on Cortical Sheet\n"
        "(bright = high average activation for that class; right col = winner-takes-all selectivity)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig6_activation_heatmaps.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 7: Minority Class Robustness vs. Imbalance Ratio
# ═════════════════════════════════════════════════════════════════════════════

def plot_minority_robustness(
    results: dict,
    out_dir: Path = OUTPUT_DIR / "figures",
) -> None:
    """For each class, plot accuracy vs. that class's training proportion.

    Coloured lines connect the four variants.  Shows whether topo models
    maintain minority-class accuracy when the class is underrepresented.
    """
    # Build mapping: class → {variant → [(proportion, accuracy), ...]}
    data: dict[int, dict[str, list]] = {c: {v: [] for v in VARIANT_ORDER}
                                         for c in range(N_CLASSES)}

    prop_map = {
        "balanced":          [0.25, 0.25, 0.25, 0.25],
        "top_dominant":      [0.70, 0.10, 0.10, 0.10],
        "bottom_dominant":   [0.10, 0.70, 0.10, 0.10],
        "footwear_dominant": [0.10, 0.10, 0.70, 0.10],
        "bag_dominant":      [0.10, 0.10, 0.10, 0.70],
        "extreme_skew":      [0.55, 0.25, 0.15, 0.05],
    }

    for diet, proportions in prop_map.items():
        for vi, variant in enumerate(VARIANT_ORDER):
            r = _get(results, diet, variant)
            if r is None:
                continue
            for ci in range(N_CLASSES):
                data[ci][variant].append(
                    (proportions[ci], r["final_val_acc_per_class"][ci])
                )

    fig, axes = plt.subplots(1, N_CLASSES, figsize=(16, 4),
                             gridspec_kw={"wspace": 0.28})

    for ci, ax in enumerate(axes):
        for variant in VARIANT_ORDER:
            pts = sorted(data[ci][variant], key=lambda x: x[0])
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "o-",
                    color=VARIANT_COLORS[variant],
                    label=VARIANT_LABELS[variant],
                    linewidth=2, markersize=5, alpha=0.85)

        ax.set_title(CLASS_NAMES[ci], fontsize=10,
                     color=CLASS_HEX[CLASS_NAMES[ci]], fontweight="bold")
        ax.set_xlabel("Training proportion of this class", fontsize=8)
        ax.set_ylabel("Validation accuracy (%)" if ci == 0 else "", fontsize=8)
        ax.set_xlim(-0.02, 0.75)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=6, framealpha=0.8)
        ax.axvline(0.25, color="#999999", linestyle=":", linewidth=1,
                   label="Balanced")
        ax.grid(alpha=0.3, linewidth=0.5)

    fig.suptitle(
        "Minority Class Accuracy vs. Training Proportion\n"
        "(each point = one training diet; lower proportion = more imbalanced)",
        fontsize=11,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig7_minority_robustness.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 8: TopoLoss value vs. Diet — does imbalance hurt topo organisation?
# ═════════════════════════════════════════════════════════════════════════════

def plot_topo_loss_vs_diet(
    results: dict,
    out_dir: Path = OUTPUT_DIR / "figures",
) -> None:
    """Final training TopoLoss for each variant across diets."""
    n_v  = len(VARIANT_ORDER)
    n_d  = len(DIET_ORDER)
    bar_w = 0.2
    fig, ax = plt.subplots(figsize=(13, 4))

    x_centers = np.arange(n_d)

    for vi, variant in enumerate(VARIANT_ORDER):
        if variant == "baseline":
            continue
        vals = []
        for diet in DIET_ORDER:
            r = _get(results, diet, variant)
            if r is not None:
                topo_hist = r["history"]["topo"]
                vals.append(topo_hist[-1] if topo_hist else 0.0)
            else:
                vals.append(0.0)

        x_pos = x_centers + (vi - (n_v - 1) / 2) * bar_w
        ax.bar(x_pos, vals, bar_w,
               color=VARIANT_COLORS[variant],
               label=VARIANT_LABELS[variant],
               alpha=0.82, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([DIET_SHORT[d] for d in DIET_ORDER], fontsize=8)
    ax.set_ylabel("Final TopoLoss (Laplacian Pyramid)", fontsize=10)
    ax.set_xlabel("Training diet", fontsize=9)
    ax.set_title(
        "Topographic Loss at End of Training — Effect of Imbalanced Diets\n"
        "(lower = stronger spatial smoothness of weight vectors)",
        fontsize=11,
    )
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig8_topo_loss_vs_diet.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def get_config() -> dict:
    p = argparse.ArgumentParser(
        description="Generate figures for the imbalanced diet topographic experiment."
    )
    default_res = str(OUTPUT_DIR / "results" / "imbalanced_diet_results_latest.json")
    p.add_argument("--results",    default=default_res,
                   help=f"Path to results JSON (default: {default_res})")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR / "figures"),
                   help="Directory to write figure PNGs")
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--no-blended", action="store_true",
                   help="Skip blended cortical maps in Figure 1")
    cli = p.parse_args()
    return {
        "results":     cli.results,
        "output_dir":  cli.output_dir,
        "hidden_size": cli.hidden_size,
        "show_blended": not cli.no_blended,
    }


def main() -> None:
    cfg = get_config()
    results_path = cfg["results"]
    out_dir      = Path(cfg["output_dir"])
    hidden_size  = cfg["hidden_size"]

    print(f"Loading results from: {results_path}")
    results = load_results(results_path)
    print(f"Found {len(results)} run records.\n")

    print("Generating figures …")

    print("  Figure 1: Cortical selectivity atlas …")
    plot_cortical_atlas(results, hidden_size=hidden_size,
                        out_dir=out_dir, show_blended=cfg["show_blended"])

    print("  Figure 2: Class territory stacked bars …")
    plot_class_territories(results, out_dir=out_dir)

    print("  Figure 3: Spatial clustering scores …")
    plot_spatial_clustering(results, out_dir=out_dir)

    print("  Figure 4: Per-class accuracy heatmaps …")
    plot_per_class_accuracy_heatmaps(results, out_dir=out_dir)

    print("  Figure 5: Training loss curves …")
    plot_training_curves(results, out_dir=out_dir)

    print("  Figure 6: Per-class activation heatmaps …")
    plot_activation_heatmaps(results, hidden_size=hidden_size, out_dir=out_dir)

    print("  Figure 7: Minority class robustness …")
    plot_minority_robustness(results, out_dir=out_dir)

    print("  Figure 8: TopoLoss vs diet …")
    plot_topo_loss_vs_diet(results, out_dir=out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
