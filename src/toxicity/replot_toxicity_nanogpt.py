"""
replot_toxicity_nanogpt.py
──────────────────────────
Re-generate all plots from previously saved JSON results without re-running
any model evaluations.  Run this after tweaking plot code.

Reads from outputs/toxicity_nanogpt/ (or --output_dir):
  results.json                  summary toxicity + PPL per model
  pruning_tau{N_N}.json         per-model pruning sweep (one per tau value)
  t_stats_tau{N_N}.json         per-layer neuron t-statistics (one per tau)
  svd_selectivity_tau{N_N}.json per-layer SVD selectivity + effective rank
  svd_pruning_tau{N_N}.json     per-model SVD-direction pruning sweep

Writes to the same directory (overwriting existing PNGs):
  toxicity_comparison.png
  toxicity_ppl_scatter.png
  per_prompt_heatmap.png        (skipped unless per_completion_toxicity present)
  toxicity_multidim.png
  pruning_comparison.png
  effective_rank_per_layer.png
  effective_rank_mean.png
  singular_value_spectra.png
  svd_vs_neuron_pruning.png
  selectivity/{label}/
    t_stat_distribution.png
    per_layer_concentration.png
    cortical_sheet_selectivity.png
    cortical_sheet_mean_abs_t.png
    pruning_curves.png

Usage:
  python src/test/replot_toxicity_nanogpt.py
  python src/test/replot_toxicity_nanogpt.py --output_dir outputs/toxicity_nanogpt
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]

# ── Shared constants (must stay in sync with eval_toxicity_nanogpt.py) ─────────
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cortical_sheet_dims(n: int) -> tuple[int, int]:
    """Most square-like (H, W) factorisation of n."""
    best_h, best_w = 1, n
    for h in range(2, int(math.isqrt(n)) + 1):
        if n % h == 0:
            best_h, best_w = h, n // h
    return best_h, best_w


def _load_json(path: Path) -> dict | None:
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return None


# ── Cross-model comparison plots ───────────────────────────────────────────────

def plot_comparison(results: dict, output_dir: Path) -> None:
    labels = list(results.keys())
    means  = [r["toxicity_scores"]["toxicity"]["mean"] for r in results.values()]
    maxs   = [r["toxicity_scores"]["toxicity"]["max"]  for r in results.values()]
    p95s   = [r["toxicity_scores"]["toxicity"]["p95"]  for r in results.values()]
    ppls   = [r["perplexity"]                           for r in results.values()]

    x     = np.arange(len(labels))
    width = 0.25

    # ── 1. Bar chart: mean / p95 / max toxicity ────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    bars_mean = ax.bar(x - width, means, width, label="Mean",
                       color=[COLORS[i % len(COLORS)] for i in range(len(labels))], alpha=0.9)
    ax.bar(x,         p95s,  width, label="p95",
           color=[COLORS[i % len(COLORS)] for i in range(len(labels))], alpha=0.6)
    ax.bar(x + width, maxs,  width, label="Max",
           color=[COLORS[i % len(COLORS)] for i in range(len(labels))], alpha=0.3)

    for bar, v in zip(bars_mean, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Toxicity score")
    ax.set_title("NanoGPT Toxicity Benchmark — Regular vs. Topo-regularised")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, min(1.0, max(maxs) * 1.15))
    plt.tight_layout()
    p = output_dir / "toxicity_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    # ── 2. Scatter: toxicity vs. PPL ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (label, m, ppl) in enumerate(zip(labels, means, ppls)):
        ax.scatter(ppl, m, s=120, color=COLORS[i % len(COLORS)], zorder=3, label=label)
        ax.annotate(label, (ppl, m), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Perplexity (lower = better language model)")
    ax.set_ylabel("Mean Toxicity (lower = safer)")
    ax.set_title("Toxicity vs. Perplexity trade-off")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = output_dir / "toxicity_ppl_scatter.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    # ── 3. Per-prompt heatmap (only if per_completion_toxicity is present) ──
    all_per = [r.get("per_completion_toxicity", []) for r in results.values()]
    if all_per and all(len(a) > 0 and len(a) == len(all_per[0]) for a in all_per):
        n_prompts = len(all_per[0])
        matrix    = np.array(all_per)
        fig, ax   = plt.subplots(figsize=(max(12, n_prompts // 5), max(4, len(labels))))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Prompt index")
        ax.set_title("Per-prompt toxicity scores")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        plt.tight_layout()
        p = output_dir / "per_prompt_heatmap.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {p}")

    # ── 4. Multi-dimension toxicity bar chart ────────────────────────────────
    dims = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]
    baseline_key = list(results.keys())[0]
    if all(dim in results[baseline_key]["toxicity_scores"] for dim in dims):
        fig, ax = plt.subplots(figsize=(len(dims) * 1.5, 5))
        x_pos   = np.arange(len(dims))
        bw      = 0.8 / len(labels)
        for i, (label, r) in enumerate(results.items()):
            vals = [r["toxicity_scores"].get(d, {}).get("mean", 0.0) for d in dims]
            ax.bar(x_pos + i * bw, vals, bw,
                   label=label, color=COLORS[i % len(COLORS)], alpha=0.85)
        ax.set_xticks(x_pos + bw * (len(labels) - 1) / 2)
        ax.set_xticklabels([d.replace("_", "\n") for d in dims], fontsize=9)
        ax.set_ylabel("Mean score")
        ax.set_title("Multi-dimension toxicity comparison")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        p = output_dir / "toxicity_multidim.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {p}")


def plot_pruning_comparison(pruning_results: dict[str, dict], output_dir: Path) -> None:
    """Plot raw toxicity + raw PPL curves for all models in one figure."""
    valid = {k: v for k, v in pruning_results.items() if v and v.get("pruning_fractions")}
    if not valid:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    for i, (label, tp) in enumerate(valid.items()):
        fracs        = tp["pruning_fractions"]
        baseline_tox = tp["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
        pruned_tox   = [tp["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in fracs]
        baseline_ppl = tp["unpruned"]["ppl"]
        ppls         = [tp["pruned"][str(f)]["perplexity"] for f in fracs]
        baseline_vl  = tp["unpruned"]["val_loss"]
        vls          = [tp["pruned"][str(f)]["val_loss"] for f in fracs]
        x_pct        = [f * 100 for f in fracs]
        color        = COLORS[i % len(COLORS)]

        ax1.plot([0] + x_pct, [baseline_tox] + pruned_tox, "o-", label=label, color=color, linewidth=2)
        ax2.plot([0] + x_pct, [baseline_ppl] + ppls,       "s-", label=label, color=color, linewidth=2)
        ax3.plot([0] + x_pct, [baseline_vl]  + vls,        "^-", label=label, color=color, linewidth=2)

    ax2.set_yscale("log")
    ax3.set_yscale("log")
    for ax, ylabel, title in [
        (ax1, "Toxicity (mean)", "Toxicity Reduction via Neuron Pruning"),
        (ax2, "Perplexity",     "Perplexity Cost of Neuron Pruning"),
        (ax3, "Val Loss",       "Validation Loss Cost of Neuron Pruning"),
    ]:
        ax.set_xlabel("Neurons Pruned (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Neuron Pruning — All Models", fontsize=13)
    plt.tight_layout()
    p = output_dir / "pruning_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    # ── Toxic-neuron concentration bar chart across models ───────────────────
    ns_pairs = [
        (label, tp.get("neuron_stats", {}))
        for label, tp in valid.items()
    ]
    frac_sigs = [ns.get("frac_significant_t2", 0.0) for _, ns in ns_pairs]
    if any(f > 0 for f in frac_sigs):
        names = [n for n, _ in ns_pairs]
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
        x_pos = np.arange(len(names))
        bars  = ax.bar(x_pos, [f * 100 for f in frac_sigs],
                       color=[COLORS[i % len(COLORS)] for i in range(len(names))])
        for bar, f in zip(bars, frac_sigs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{f * 100:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Toxic-Selective Neurons (%)")
        ax.set_title("Fraction of Neurons with t > 2 (toxic vs. non-toxic)")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        p = output_dir / "toxicity_neuron_concentration.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {p}")


def plot_global_pruning_comparison(
    global_pruning_results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Raw toxicity + raw PPL for all models under global neuron pruning."""
    valid = {k: v for k, v in global_pruning_results.items()
             if v and v.get("global_pruning_fractions")}
    if not valid:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    for i, (label, tp) in enumerate(valid.items()):
        fracs        = tp["global_pruning_fractions"]
        baseline_tox = tp["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
        pruned_tox   = [tp["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in fracs]
        baseline_ppl = tp["unpruned"]["ppl"]
        ppls         = [tp["pruned"][str(f)]["perplexity"] for f in fracs]
        baseline_vl  = tp["unpruned"]["val_loss"]
        vls          = [tp["pruned"][str(f)]["val_loss"] for f in fracs]
        x_pct        = [f * 100 for f in fracs]
        color        = COLORS[i % len(COLORS)]
        ax1.plot([0] + x_pct, [baseline_tox] + pruned_tox, "o-", label=label, color=color, linewidth=2)
        ax2.plot([0] + x_pct, [baseline_ppl] + ppls,       "s-", label=label, color=color, linewidth=2)
        ax3.plot([0] + x_pct, [baseline_vl]  + vls,        "^-", label=label, color=color, linewidth=2)

    ax2.set_yscale("log")
    ax3.set_yscale("log")
    for ax, ylabel, title in [
        (ax1, "Toxicity (mean)", "Toxicity Reduction via Global Neuron Pruning"),
        (ax2, "Perplexity",     "Perplexity Cost of Global Neuron Pruning"),
        (ax3, "Val Loss",       "Validation Loss Cost of Global Neuron Pruning"),
    ]:
        ax.set_xlabel("Global Neurons Pruned (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Global Neuron Pruning — All Models", fontsize=13)
    plt.tight_layout()
    p = output_dir / "global_pruning_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ── Per-model selectivity plots ────────────────────────────────────────────────

def save_selectivity_visualizations(
    t_stats_per_layer: dict[int, np.ndarray],
    global_stats:      dict,
    pruning_result:    dict,
    label:             str,
    vis_dir:           Path,
) -> None:
    vis_dir.mkdir(parents=True, exist_ok=True)

    n_layers = len(t_stats_per_layer)
    if n_layers == 0:
        return

    # ── 1. t-statistic distribution ─────────────────────────────────────────
    ncols = 4
    nrows = (n_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for i in range(n_layers):
        ax = axes_flat[i]
        t  = t_stats_per_layer[i]
        ax.hist(t, bins=60, color=COLORS[i % len(COLORS)], alpha=0.75)
        sig_frac = float((t > 2.0).mean()) * 100
        ax.axvline(2.0,  color="red",  linestyle="--", linewidth=1)
        ax.axvline(-2.0, color="blue", linestyle="--", linewidth=1)
        ax.set_title(f"Layer {i}  (t>2: {sig_frac:.1f}%)", fontsize=10)
        ax.set_xlabel("t-stat", fontsize=8)
        ax.set_ylabel("# neurons", fontsize=8)
    for j in range(n_layers, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle(f"{label} — MLP neuron t-statistic distributions", fontsize=13)
    plt.tight_layout()
    p = vis_dir / "t_stat_distribution.png"
    fig.savefig(p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"      → {p}")

    # ── 2. Per-layer toxic-neuron concentration ──────────────────────────────
    fracs_t2  = [float((t_stats_per_layer[i] > 2.0).mean()) * 100 for i in range(n_layers)]
    fracs_tn2 = [float((t_stats_per_layer[i] < -2.0).mean()) * 100 for i in range(n_layers)]
    x_pos = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(max(8, n_layers), 4))
    ax.bar(x_pos - 0.2, fracs_t2,  0.4, label="t > +2 (toxic-selective)",     color="#d62728", alpha=0.85)
    ax.bar(x_pos + 0.2, fracs_tn2, 0.4, label="t < -2 (non-toxic-selective)", color="#1f77b4", alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
    ax.set_ylabel("% of neurons")
    ax.set_title(f"{label} — Toxic-selective neuron concentration per layer")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p = vis_dir / "per_layer_concentration.png"
    fig.savefig(p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"      → {p}")

    # ── 3. Cortical-sheet selectivity maps ───────────────────────────────────
    n_neurons = len(t_stats_per_layer[0])
    H, W = _cortical_sheet_dims(n_neurons)
    ncols = 4
    nrows = (n_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    t_all_arr = np.array([t_stats_per_layer[i] for i in range(n_layers)])
    vmax = float(np.percentile(np.abs(t_all_arr), 97))

    for i in range(n_layers):
        ax = axes_flat[i]
        sheet = t_stats_per_layer[i].reshape(H, W)
        im = ax.imshow(sheet, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"Layer {i}", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    for j in range(n_layers, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle(
        f"{label} — Cortical-sheet selectivity (t-stat, red=toxic, blue=non-toxic)\n"
        f"Sheet: {H}×{W}, total neurons/layer: {n_neurons}",
        fontsize=12,
    )
    plt.tight_layout()
    p = vis_dir / "cortical_sheet_selectivity.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"      → {p}")

    # ── 4. Mean |t| across layers on the cortical sheet ─────────────────────
    mean_abs_t = np.stack([np.abs(t_stats_per_layer[i]) for i in range(n_layers)]).mean(0)
    H2, W2 = _cortical_sheet_dims(len(mean_abs_t))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mean_abs_t.reshape(H2, W2), cmap="hot", aspect="auto")
    ax.set_title(f"{label} — Mean |t| across layers ({H2}×{W2} sheet)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    plt.tight_layout()
    p = vis_dir / "cortical_sheet_mean_abs_t.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"      → {p}")

    # ── 5. Pruning curves ────────────────────────────────────────────────────
    fracs = pruning_result.get("pruning_fractions", [])
    if fracs:
        baseline_tox = pruning_result["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
        pruned_tox   = [pruning_result["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"]
                        for f in fracs]
        baseline_ppl = pruning_result["unpruned"]["ppl"]
        ppls         = [pruning_result["pruned"][str(f)]["perplexity"] for f in fracs]
        baseline_vl  = pruning_result["unpruned"]["val_loss"]
        vls          = [pruning_result["pruned"][str(f)]["val_loss"] for f in fracs]
        norm_tox     = [v / max(baseline_tox, 1e-8) for v in pruned_tox]
        x_pct        = [f * 100 for f in fracs]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        ax1.plot([0] + x_pct, [1.0] + norm_tox,          "o-", color="#d62728", linewidth=2)
        ax2.plot([0] + x_pct, [baseline_ppl] + ppls,      "s-", color="#1f77b4", linewidth=2)
        ax3.plot([0] + x_pct, [baseline_vl]  + vls,       "^-", color="#2ca02c", linewidth=2)
        ax2.set_yscale("log")
        ax3.set_yscale("log")
        ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        for ax, ylabel, title in [
            (ax1, "Toxicity (relative to unpruned)", "Toxicity reduction vs. pruning"),
            (ax2, "Perplexity", "PPL cost of pruning"),
            (ax3, "Val Loss",   "Val loss cost of pruning"),
        ]:
            ax.set_xlabel("Neurons pruned (%)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"{label} — Toxicity pruning sweep", fontsize=13)
        plt.tight_layout()
        p = vis_dir / "pruning_curves.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"      → {p}")

    # ── 6. Pruned-neuron cortical-sheet masks ────────────────────────────────
    fracs = pruning_result.get("pruning_fractions", [])
    if fracs:
        n_neurons_sheet = len(t_stats_per_layer[0])
        Hm, Wm = _cortical_sheet_dims(n_neurons_sheet)

        for frac in fracs:
            n_prune = max(1, int(n_neurons_sheet * frac))
            ncols_m = 4
            nrows_m = (n_layers + ncols_m - 1) // ncols_m
            fig, axes_m = plt.subplots(nrows_m, ncols_m,
                                       figsize=(4 * ncols_m, 4 * nrows_m))
            axes_flat_m = list(axes_m.flat) if hasattr(axes_m, "flat") else [axes_m]

            t_all_arr_m = np.array([t_stats_per_layer[i] for i in range(n_layers)])
            vmax_m = float(np.percentile(np.abs(t_all_arr_m), 97))

            for i in range(n_layers):
                ax = axes_flat_m[i]
                t = t_stats_per_layer[i]

                sheet = t.reshape(Hm, Wm).astype(float)
                ax.imshow(sheet, cmap="RdBu_r", vmin=-vmax_m, vmax=vmax_m,
                          aspect="auto", interpolation="nearest")

                pruned_idx = np.argsort(t)[-n_prune:]
                mask = np.zeros(n_neurons_sheet, dtype=float)
                mask[pruned_idx] = 1.0
                overlay = np.zeros((Hm, Wm, 4), dtype=float)
                overlay[..., 0] = 1.0
                overlay[..., 1] = 0.95
                overlay[..., 2] = 0.0
                overlay[..., 3] = mask.reshape(Hm, Wm) * 0.85
                ax.imshow(overlay, aspect="auto", interpolation="nearest")

                pct_pruned = n_prune / n_neurons_sheet * 100
                ax.set_title(f"L{i}  ({pct_pruned:.1f}% pruned)", fontsize=9)
                ax.axis("off")

            for j in range(n_layers, len(axes_flat_m)):
                axes_flat_m[j].axis("off")

            fig.suptitle(
                f"{label} — Pruned neurons on cortical sheet  |  "
                f"fraction={frac:.0%}  ({n_prune}/{n_neurons_sheet} per layer)\n"
                f"Background: t-stat (red=toxic)   Overlay: pruned neurons (yellow)",
                fontsize=11,
            )
            plt.tight_layout()
            frac_str = f"{int(frac * 100):02d}"
            p = vis_dir / f"pruned_neurons_cortical_{frac_str}pct.png"
            fig.savefig(p, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"      → {p}")


# ── Effective rank + SVD plots ─────────────────────────────────────────────────

def plot_effective_rank(
    svd_results: dict[str, dict[str, dict]],
    output_dir:  Path,
) -> None:
    """Effective rank per layer, mean bar chart, and singular value spectra."""
    if not svd_results:
        return

    labels   = list(svd_results.keys())
    n_layers = max(max(int(li) for li in v) + 1 for v in svd_results.values())

    # ── 1. Effective rank per layer ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, n_layers), 5))
    x_pos = np.arange(n_layers)
    for i, label in enumerate(labels):
        ranks = [
            svd_results[label].get(str(li), {}).get("effective_rank", float("nan"))
            for li in range(n_layers)
        ]
        ax.plot(x_pos, ranks, "o-", label=label, color=COLORS[i % len(COLORS)], linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective rank (↑ more expressive)")
    ax.set_title("MLP c_proj effective rank per layer — Regular vs. Topo nanoGPT")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = output_dir / "effective_rank_per_layer.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    # ── 2. Mean effective rank bar chart ─────────────────────────────────────
    mean_ranks = [
        float(np.nanmean([
            svd_results[label].get(str(li), {}).get("effective_rank", float("nan"))
            for li in range(n_layers)
        ]))
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    bars = ax.bar(range(len(labels)), mean_ranks,
                  color=[COLORS[i % len(COLORS)] for i in range(len(labels))])
    for bar, v in zip(bars, mean_ranks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean effective rank across layers")
    ax.set_title("Mean MLP effective rank — Regular vs. Topo nanoGPT")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p = output_dir / "effective_rank_mean.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    # ── 3. Singular value spectra (median across layers per model) ───────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, label in enumerate(labels):
        all_sv = [
            svd_results[label][li]["singular_values"]
            for li in svd_results[label]
            if "singular_values" in svd_results[label][li]
        ]
        if not all_sv:
            continue
        min_len = min(len(sv) for sv in all_sv)
        arr = np.array([sv[:min_len] for sv in all_sv])
        median_sv = np.median(arr, axis=0)
        energy = (median_sv ** 2).sum()
        frac   = (median_sv ** 2) / max(energy, 1e-12)
        ax.plot(np.cumsum(frac), label=label, color=COLORS[i % len(COLORS)], linewidth=2)
    ax.set_xlabel("Number of singular components")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("Singular value spectrum — c_proj weight (median across layers)")
    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.5, label="90% variance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = output_dir / "singular_value_spectra.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def plot_svd_pruning_comparison(
    svd_pruning_results:    dict[str, dict],
    neuron_pruning_results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Compare neuron-level vs SVD-space pruning on toxicity and PPL."""
    valid_svd  = {k: v for k, v in svd_pruning_results.items()  if v and v.get("pruning_fractions")}
    valid_neur = {k: v for k, v in neuron_pruning_results.items() if v and v.get("pruning_fractions")}
    if not valid_svd and not valid_neur:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2  = axes
    color_idx = 0

    for label, tp in valid_neur.items():
        fracs      = tp["pruning_fractions"]
        base_tox   = tp["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
        pruned_tox = [tp["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in fracs]
        baseline_ppl = tp["unpruned"]["ppl"]
        ppls         = [tp["pruned"][str(f)]["perplexity"] for f in fracs]
        norm_tox     = [v / max(base_tox, 1e-8) for v in pruned_tox]
        x_pct        = [f * 100 for f in fracs]
        c = COLORS[color_idx % len(COLORS)]
        ax1.plot([0] + x_pct, [1.0] + norm_tox,      "o-", color=c, linewidth=2,
                 label=f"{label} (neuron)")
        ax2.plot([0] + x_pct, [baseline_ppl] + ppls, "o-", color=c, linewidth=2,
                 label=f"{label} (neuron)")
        color_idx += 1

    for label, tp in valid_svd.items():
        fracs        = tp["pruning_fractions"]
        base_tox     = tp["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
        pruned_tox   = [tp["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in fracs]
        baseline_ppl = tp["unpruned"]["ppl"]
        ppls         = [tp["pruned"][str(f)]["perplexity"] for f in fracs]
        norm_tox     = [v / max(base_tox, 1e-8) for v in pruned_tox]
        x_pct        = [f * 100 for f in fracs]
        c = COLORS[color_idx % len(COLORS)]
        ax1.plot([0] + x_pct, [1.0] + norm_tox,      "s--", color=c, linewidth=2,
                 label=f"{label} (SVD)")
        ax2.plot([0] + x_pct, [baseline_ppl] + ppls, "s--", color=c, linewidth=2,
                 label=f"{label} (SVD)")
        color_idx += 1

    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    for ax, ylabel, title in [
        (ax1, "Toxicity (fraction of unpruned)", "Toxicity reduction: neuron vs SVD pruning"),
        (ax2, "Perplexity", "PPL cost: neuron vs SVD pruning"),
    ]:
        ax.set_xlabel("Components / neurons pruned (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    ax2.set_yscale("log")
    fig.suptitle("Neuron pruning vs. SVD-direction pruning", fontsize=13)
    plt.tight_layout()
    p = output_dir / "svd_vs_neuron_pruning.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def plot_svd_cross_model_comparison(
    svd_pruning_results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Raw toxicity + raw PPL for all SVD-pruned models on one figure."""
    valid = {k: v for k, v in svd_pruning_results.items() if v and v.get("pruning_fractions")}
    if not valid:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, (label, tp) in enumerate(valid.items()):
        fracs        = tp["pruning_fractions"]
        baseline_tox = tp["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
        pruned_tox   = [tp["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in fracs]
        baseline_ppl = tp["unpruned"]["ppl"]
        ppls         = [tp["pruned"][str(f)]["perplexity"] for f in fracs]
        x_pct        = [f * 100 for f in fracs]
        color        = COLORS[i % len(COLORS)]
        ax1.plot([0] + x_pct, [baseline_tox] + pruned_tox, "o-", label=label, color=color, linewidth=2)
        ax2.plot([0] + x_pct, [baseline_ppl] + ppls,       "s-", label=label, color=color, linewidth=2)

    ax2.set_yscale("log")
    for ax, ylabel, title in [
        (ax1, "Toxicity (mean)", "Toxicity Reduction via SVD Pruning"),
        (ax2, "Perplexity",     "Perplexity Cost of SVD Pruning"),
    ]:
        ax.set_xlabel("SVD Components Pruned (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("SVD Pruning — All Models", fontsize=13)
    plt.tight_layout()
    p = output_dir / "svd_pruning_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def plot_amplification_comparison(
    amp_results:     dict[str, dict],
    pruning_results: dict[str, dict],
    output_dir:      Path,
) -> None:
    """Cross-model toxicity + PPL curves for amplified neurons, plus a combined
    pruning-vs-amplification overlay across all models."""
    valid_amp  = {k: v for k, v in amp_results.items()  if v and v.get("amp_fracs")}
    valid_prun = {k: v for k, v in pruning_results.items() if v and v.get("pruning_fractions")}
    if not valid_amp:
        return

    # ── Cross-model amplification-only plot ──────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, (label, ta) in enumerate(valid_amp.items()):
        fracs      = ta["amp_fracs"]
        amp_factor = ta["amp_factor"]
        bt         = ta["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
        bp         = ta["unpruned"]["ppl"]
        a_tox      = [ta["amplified"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in fracs]
        a_ppls     = [ta["amplified"][str(f)]["perplexity"] for f in fracs]
        color      = COLORS[i % len(COLORS)]
        ax1.plot([0] + [f * 100 for f in fracs], [bt] + a_tox,
                 "o-", label=f"{label} (×{amp_factor})", color=color, linewidth=2)
        ax2.plot([0] + [f * 100 for f in fracs], [bp] + a_ppls,
                 "s-", label=f"{label} (×{amp_factor})", color=color, linewidth=2)
    ax2.set_yscale("log")
    for ax, ylabel, title in [
        (ax1, "Toxicity (mean)", "Toxicity Increase via Amplification"),
        (ax2, "Perplexity",     "Perplexity Cost of Amplification"),
    ]:
        ax.set_xlabel("Neurons Amplified (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Toxic-neuron Amplification — All Models", fontsize=13)
    plt.tight_layout()
    p = output_dir / "amplification_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    # ── Combined pruning + amplification overlay ─────────────────────────────
    all_labels = sorted(set(list(valid_amp.keys()) + list(valid_prun.keys())))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, label in enumerate(all_labels):
        color = COLORS[i % len(COLORS)]
        if label in valid_prun:
            tp      = valid_prun[label]
            p_fracs = tp["pruning_fractions"]
            bt      = tp["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
            bp      = tp["unpruned"]["ppl"]
            p_tox   = [tp["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in p_fracs]
            p_ppls  = [tp["pruned"][str(f)]["perplexity"] for f in p_fracs]
            ax1.plot([0] + [f * 100 for f in p_fracs], [bt] + p_tox,
                     "o--", color=color, linewidth=2, alpha=0.8, label=f"{label} (prune)")
            ax2.plot([0] + [f * 100 for f in p_fracs], [bp] + p_ppls,
                     "s--", color=color, linewidth=2, alpha=0.8, label=f"{label} (prune)")
        if label in valid_amp:
            ta      = valid_amp[label]
            a_fracs = ta["amp_fracs"]
            af      = ta["amp_factor"]
            bt      = ta["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
            bp      = ta["unpruned"]["ppl"]
            a_tox   = [ta["amplified"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in a_fracs]
            a_ppls  = [ta["amplified"][str(f)]["perplexity"] for f in a_fracs]
            ax1.plot([0] + [f * 100 for f in a_fracs], [bt] + a_tox,
                     "o-", color=color, linewidth=2, label=f"{label} (amp×{af})")
            ax2.plot([0] + [f * 100 for f in a_fracs], [bp] + a_ppls,
                     "s-", color=color, linewidth=2, label=f"{label} (amp×{af})")
    ax2.set_yscale("log")
    for ax, ylabel, title in [
        (ax1, "Toxicity (mean)", "Pruning vs. Amplification — Toxicity"),
        (ax2, "Perplexity",     "Pruning vs. Amplification — PPL"),
    ]:
        ax.set_xlabel("Neurons selected (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Pruning vs. Amplification of Toxic Neurons", fontsize=13)
    plt.tight_layout()
    p = output_dir / "prune_vs_amplify_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def plot_attenuation_comparison(
    att_results:     dict[str, dict],
    pruning_results: dict[str, dict],
    output_dir:      Path,
) -> None:
    """Cross-model toxicity + PPL + val-loss for attenuated neurons,
    plus a combined pruning-vs-attenuation overlay across all models."""
    valid_att  = {k: v for k, v in att_results.items()  if v and v.get("att_fracs")}
    valid_prun = {k: v for k, v in pruning_results.items() if v and v.get("pruning_fractions")}
    if not valid_att:
        return

    # ── Cross-model attenuation-only plot ────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    for i, (label, ta) in enumerate(valid_att.items()):
        fracs      = ta["att_fracs"]
        att_factor = ta["att_factor"]
        bt         = ta["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
        bp         = ta["unpruned"]["ppl"]
        bv         = ta["unpruned"].get("val_loss", float("nan"))
        a_tox  = [ta["attenuated"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in fracs]
        a_ppls = [ta["attenuated"][str(f)]["perplexity"] for f in fracs]
        a_vls  = [ta["attenuated"][str(f)].get("val_loss", float("nan")) for f in fracs]
        color  = COLORS[i % len(COLORS)]
        x_pct  = [f * 100 for f in fracs]
        ax1.plot([0] + x_pct, [bt] + a_tox,  "o-", label=f"{label} (×{att_factor})", color=color, linewidth=2)
        ax2.plot([0] + x_pct, [bp] + a_ppls, "s-", label=f"{label} (×{att_factor})", color=color, linewidth=2)
        ax3.plot([0] + x_pct, [bv] + a_vls,  "^-", label=f"{label} (×{att_factor})", color=color, linewidth=2)
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    for ax, ylabel, title, xlabel in [
        (ax1, "Toxicity (mean)", "Toxicity Reduction via Attenuation",  "Neurons Attenuated (%)"),
        (ax2, "Perplexity",     "Perplexity Cost of Attenuation",       "Neurons Attenuated (%)"),
        (ax3, "Val Loss",       "Val Loss Cost of Attenuation",         "Neurons Attenuated (%)"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Toxic-neuron Attenuation — All Models", fontsize=13)
    plt.tight_layout()
    p = output_dir / "attenuation_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    # ── Combined pruning + attenuation overlay ───────────────────────────────
    all_labels = sorted(set(list(valid_att.keys()) + list(valid_prun.keys())))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, label in enumerate(all_labels):
        color = COLORS[i % len(COLORS)]
        if label in valid_prun:
            tp      = valid_prun[label]
            p_fracs = tp["pruning_fractions"]
            bt      = tp["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
            bp      = tp["unpruned"]["ppl"]
            p_tox   = [tp["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in p_fracs]
            p_ppls  = [tp["pruned"][str(f)]["perplexity"] for f in p_fracs]
            ax1.plot([0] + [f * 100 for f in p_fracs], [bt] + p_tox,
                     "o--", color=color, linewidth=2, alpha=0.8, label=f"{label} (prune)")
            ax2.plot([0] + [f * 100 for f in p_fracs], [bp] + p_ppls,
                     "s--", color=color, linewidth=2, alpha=0.8, label=f"{label} (prune)")
        if label in valid_att:
            ta      = valid_att[label]
            a_fracs = ta["att_fracs"]
            af      = ta["att_factor"]
            bt      = ta["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
            bp      = ta["unpruned"]["ppl"]
            a_tox   = [ta["attenuated"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in a_fracs]
            a_ppls  = [ta["attenuated"][str(f)]["perplexity"] for f in a_fracs]
            ax1.plot([0] + [f * 100 for f in a_fracs], [bt] + a_tox,
                     "o-", color=color, linewidth=2, label=f"{label} (att×{af})")
            ax2.plot([0] + [f * 100 for f in a_fracs], [bp] + a_ppls,
                     "s-", color=color, linewidth=2, label=f"{label} (att×{af})")
    ax2.set_yscale("log")
    for ax, ylabel, title in [
        (ax1, "Toxicity (mean)", "Pruning vs. Attenuation — Toxicity"),
        (ax2, "Perplexity",     "Pruning vs. Attenuation — PPL"),
    ]:
        ax.set_xlabel("Neurons selected (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Pruning vs. Attenuation of Toxic Neurons", fontsize=13)
    plt.tight_layout()
    p = output_dir / "prune_vs_attenuate_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def plot_pca_pruning_comparison(
    pca_results:     dict[str, dict],
    pruning_results: dict[str, dict],
    output_dir:      Path,
) -> None:
    """Cross-model toxicity + PPL + val-loss for PCA-pruned models,
    plus a combined neuron-pruning-vs-PCA-pruning overlay."""
    valid_pca  = {k: v for k, v in pca_results.items()  if v and v.get("pruning_fractions")}
    valid_prun = {k: v for k, v in pruning_results.items() if v and v.get("pruning_fractions")}
    if not valid_pca:
        return

    n_pca = next(iter(valid_pca.values())).get("n_pca_components", "?")

    # ── Cross-model PCA-only plot ────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    for i, (label, tp) in enumerate(valid_pca.items()):
        fracs = tp["pruning_fractions"]
        bt    = tp["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
        bp    = tp["unpruned"]["ppl"]
        bv    = tp["unpruned"].get("val_loss", float("nan"))
        p_tox  = [tp["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in fracs]
        p_ppls = [tp["pruned"][str(f)]["perplexity"] for f in fracs]
        p_vls  = [tp["pruned"][str(f)].get("val_loss", float("nan")) for f in fracs]
        color  = COLORS[i % len(COLORS)]
        x_pct  = [f * 100 for f in fracs]
        ax1.plot([0] + x_pct, [bt] + p_tox,  "o-", label=label, color=color, linewidth=2)
        ax2.plot([0] + x_pct, [bp] + p_ppls, "s-", label=label, color=color, linewidth=2)
        ax3.plot([0] + x_pct, [bv] + p_vls,  "^-", label=label, color=color, linewidth=2)
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    for ax, ylabel, title, xlabel in [
        (ax1, "Toxicity (mean)", f"Toxicity Reduction via PCA Pruning (k={n_pca})",  "PC Fraction Removed (%)"),
        (ax2, "Perplexity",     f"Perplexity Cost of PCA Pruning (k={n_pca})",       "PC Fraction Removed (%)"),
        (ax3, "Val Loss",       f"Val Loss Cost of PCA Pruning (k={n_pca})",         "PC Fraction Removed (%)"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"PCA Pruning — All Models (k={n_pca} components)", fontsize=13)
    plt.tight_layout()
    p = output_dir / "pca_pruning_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    # ── Combined neuron pruning + PCA pruning overlay ────────────────────────
    all_labels = sorted(set(list(valid_pca.keys()) + list(valid_prun.keys())))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, label in enumerate(all_labels):
        color = COLORS[i % len(COLORS)]
        if label in valid_prun:
            tp      = valid_prun[label]
            p_fracs = tp["pruning_fractions"]
            bt      = tp["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
            bp      = tp["unpruned"]["ppl"]
            p_tox   = [tp["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in p_fracs]
            p_ppls  = [tp["pruned"][str(f)]["perplexity"] for f in p_fracs]
            ax1.plot([0] + [f * 100 for f in p_fracs], [bt] + p_tox,
                     "o--", color=color, linewidth=2, alpha=0.8, label=f"{label} (neuron)")
            ax2.plot([0] + [f * 100 for f in p_fracs], [bp] + p_ppls,
                     "s--", color=color, linewidth=2, alpha=0.8, label=f"{label} (neuron)")
        if label in valid_pca:
            tp      = valid_pca[label]
            a_fracs = tp["pruning_fractions"]
            bt      = tp["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
            bp      = tp["unpruned"]["ppl"]
            a_tox   = [tp["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in a_fracs]
            a_ppls  = [tp["pruned"][str(f)]["perplexity"] for f in a_fracs]
            ax1.plot([0] + [f * 100 for f in a_fracs], [bt] + a_tox,
                     "o-", color=color, linewidth=2, label=f"{label} (PCA)")
            ax2.plot([0] + [f * 100 for f in a_fracs], [bp] + a_ppls,
                     "s-", color=color, linewidth=2, label=f"{label} (PCA)")
    ax2.set_yscale("log")
    for ax, ylabel, title in [
        (ax1, "Toxicity (mean)", "Neuron Pruning vs. PCA Pruning — Toxicity"),
        (ax2, "Perplexity",     "Neuron Pruning vs. PCA Pruning — PPL"),
    ]:
        ax.set_xlabel("Fraction selected (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Neuron Pruning vs. PCA Pruning of Toxic Subspace", fontsize=13)
    plt.tight_layout()
    p = output_dir / "neuron_vs_pca_pruning_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def save_repeng_visualizations(
    repeng_result: dict,
    label:         str,
    vis_dir:       Path,
) -> None:
    """
    Per-model representation-engineering visualizations saved to *vis_dir*.

    Plots
    -----
    1. concept_vectors_heatmap.png   — cortical-sheet of |concept-vector| magnitude
                                       at each transformer layer
    2. top_toxic_dimensions.png      — bar chart of the top-32 residual dims with the
                                       highest mean concept-vector loading across layers
    3. layer_similarity_matrix.png   — layer×layer cosine-similarity of concept vectors
    4. repeng_curves.png             — tox + PPL + val-loss vs. steering strength α
    """
    vis_dir.mkdir(parents=True, exist_ok=True)
    raw_cv = repeng_result.get("concept_vectors", {})
    if not raw_cv:
        return

    concept_vecs = {int(k): np.array(v) for k, v in raw_cv.items()}
    n_layers = len(concept_vecs)
    if n_layers == 0:
        return
    n_embd = next(iter(concept_vecs.values())).shape[0]
    H, W   = _cortical_sheet_dims(n_embd)

    # ── 1. Cortical-sheet heatmap of |concept vector| ────────────────────────
    ncols_m = 4
    nrows_m = (n_layers + ncols_m - 1) // ncols_m
    fig, axes = plt.subplots(nrows_m, ncols_m, figsize=(4 * ncols_m, 4 * nrows_m))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for idx, (li, cv) in enumerate(sorted(concept_vecs.items())):
        ax  = axes_flat[idx]
        img = np.abs(cv).reshape(H, W)
        vmax = float(np.percentile(img, 97)) or 1.0
        ax.imshow(img, cmap="magma", vmin=0, vmax=vmax, aspect="auto",
                  interpolation="nearest")
        ax.set_title(f"L{li}  |v|", fontsize=9)
        ax.axis("off")
    for j in range(n_layers, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle(
        f"{label} — Toxicity concept-vector magnitude per residual dim\n"
        f"(brighter = dimension contributes more to the toxic direction)",
        fontsize=11,
    )
    plt.tight_layout()
    p = vis_dir / "concept_vectors_heatmap.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"      → {p}")

    # ── 2. Top-k toxic residual dimensions ───────────────────────────────────
    stacked  = np.stack([concept_vecs[i] for i in sorted(concept_vecs)], axis=0)
    mean_abs = np.abs(stacked).mean(axis=0)
    top_k    = min(32, n_embd)
    top_idx  = np.argsort(mean_abs)[-top_k:][::-1]
    top_vals = mean_abs[top_idx]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(top_k), top_vals, color="#9467bd")
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([str(d) for d in top_idx], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Residual stream dimension index")
    ax.set_ylabel("Mean |concept loading| across layers")
    ax.set_title(f"{label} — Top-{top_k} residual dims in toxicity concept direction")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p = vis_dir / "top_toxic_dimensions.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"      → {p}")

    # ── 3. Layer-to-layer concept-vector similarity ───────────────────────────
    layer_ids = sorted(concept_vecs.keys())
    n_l = len(layer_ids)
    sim = np.zeros((n_l, n_l))
    for a in range(n_l):
        for b in range(n_l):
            sim[a, b] = float(np.dot(concept_vecs[layer_ids[a]],
                                     concept_vecs[layer_ids[b]]))
    fig, ax = plt.subplots(figsize=(max(6, n_l * 0.65), max(5, n_l * 0.6)))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")
    ticks = list(range(n_l))
    labs  = [f"L{i}" for i in layer_ids]
    ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=8)
    ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=8)
    ax.set_title(f"{label} — Cosine similarity of concept vectors across layers")
    plt.tight_layout()
    p = vis_dir / "layer_similarity_matrix.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"      → {p}")

    # ── 4. Steering curves (tox + PPL + val-loss vs. α) ──────────────────────
    alphas  = repeng_result.get("alphas", [])
    steered = repeng_result.get("steered", {})
    if not alphas or not steered:
        return
    bt = repeng_result["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
    bp = repeng_result["unpruned"]["ppl"]
    bv = repeng_result["unpruned"].get("val_loss", float("nan"))
    s_tox  = [steered[str(a)]["toxicity_scores"]["toxicity"]["mean"] for a in alphas]
    s_ppls = [steered[str(a)]["perplexity"] for a in alphas]
    s_vls  = [steered[str(a)].get("val_loss", float("nan")) for a in alphas]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.plot([0] + list(alphas), [bt] + s_tox,  "o-", color="#9467bd", linewidth=2)
    ax2.plot([0] + list(alphas), [bp] + s_ppls, "s-", color="#1f77b4", linewidth=2)
    ax3.plot([0] + list(alphas), [bv] + s_vls,  "^-", color="#2ca02c", linewidth=2)
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    for ax, ylabel, title in [
        (ax1, "Mean toxicity", "Toxicity vs. steering strength α"),
        (ax2, "Perplexity",    "Perplexity vs. steering strength α"),
        (ax3, "Val Loss",      "Val Loss vs. steering strength α"),
    ]:
        ax.set_xlabel("Steering coefficient α")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{label} — Representation Engineering Steering Sweep", fontsize=13)
    plt.tight_layout()
    p = vis_dir / "repeng_curves.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"      → {p}")


def plot_repeng_comparison(
    repeng_results:  dict[str, dict],
    pruning_results: dict[str, dict],
    output_dir:      Path,
) -> None:
    """
    Cross-model comparison plots for representation-engineering steering.

    Saves
    -----
    repeng_comparison.png            — 3-panel: tox + PPL + val-loss vs. α for all models
    prune_vs_repeng_comparison.png   — 2-panel: tox-PPL tradeoff frontier (pruning dashed,
                                        rep-eng solid) and normalised tox/PPL-ratio tradeoff
    """
    valid_rep  = {k: v for k, v in repeng_results.items()  if v and v.get("alphas")}
    valid_prun = {k: v for k, v in pruning_results.items() if v and v.get("pruning_fractions")}
    if not valid_rep:
        return

    # ── Cross-model rep-eng-only plot ─────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    for i, (label, tr) in enumerate(valid_rep.items()):
        alphas = tr["alphas"]
        bt     = tr["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
        bp     = tr["unpruned"]["ppl"]
        bv     = tr["unpruned"].get("val_loss", float("nan"))
        s_tox  = [tr["steered"][str(a)]["toxicity_scores"]["toxicity"]["mean"] for a in alphas]
        s_ppls = [tr["steered"][str(a)]["perplexity"] for a in alphas]
        s_vls  = [tr["steered"][str(a)].get("val_loss", float("nan")) for a in alphas]
        color  = COLORS[i % len(COLORS)]
        ax1.plot([0] + list(alphas), [bt] + s_tox,  "o-", label=label, color=color, linewidth=2)
        ax2.plot([0] + list(alphas), [bp] + s_ppls, "s-", label=label, color=color, linewidth=2)
        ax3.plot([0] + list(alphas), [bv] + s_vls,  "^-", label=label, color=color, linewidth=2)
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    for ax, ylabel, title, xlabel in [
        (ax1, "Toxicity (mean)", "Toxicity Reduction via Rep-Eng Steering",  "Steering coefficient α"),
        (ax2, "Perplexity",     "Perplexity Cost of Rep-Eng Steering",       "Steering coefficient α"),
        (ax3, "Val Loss",       "Val Loss Cost of Rep-Eng Steering",         "Steering coefficient α"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Representation Engineering Steering — All Models", fontsize=13)
    plt.tight_layout()
    p = output_dir / "repeng_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    if not valid_prun:
        return

    # ── Toxicity-PPL tradeoff: pruning vs. rep-eng ───────────────────────────
    all_labels = sorted(set(list(valid_rep.keys()) + list(valid_prun.keys())))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, label in enumerate(all_labels):
        color = COLORS[i % len(COLORS)]
        if label in valid_prun:
            tp      = valid_prun[label]
            p_fracs = tp["pruning_fractions"]
            bt_p    = tp["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
            bp_p    = tp["unpruned"]["ppl"]
            p_tox   = [tp["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in p_fracs]
            p_ppls  = [tp["pruned"][str(f)]["perplexity"] for f in p_fracs]
            p_pr    = [tp["pruned"][str(f)]["ppl_ratio"] for f in p_fracs]
            ax1.plot([bt_p] + p_tox, [bp_p] + p_ppls,
                     "o--", color=color, linewidth=2, alpha=0.8, label=f"{label} (prune)")
            ax2.plot([1.0] + [t / max(bt_p, 1e-8) for t in p_tox], [1.0] + p_pr,
                     "o--", color=color, linewidth=2, alpha=0.8, label=f"{label} (prune)")
        if label in valid_rep:
            tr      = valid_rep[label]
            alphas  = tr["alphas"]
            bt_r    = tr["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
            bp_r    = tr["unpruned"]["ppl"]
            s_tox   = [tr["steered"][str(a)]["toxicity_scores"]["toxicity"]["mean"] for a in alphas]
            s_ppls  = [tr["steered"][str(a)]["perplexity"] for a in alphas]
            s_pr    = [tr["steered"][str(a)]["ppl_ratio"] for a in alphas]
            ax1.plot([bt_r] + s_tox, [bp_r] + s_ppls,
                     "o-", color=color, linewidth=2, label=f"{label} (repeng)")
            ax2.plot([1.0] + [t / max(bt_r, 1e-8) for t in s_tox], [1.0] + s_pr,
                     "o-", color=color, linewidth=2, label=f"{label} (repeng)")

    ax1.set_xlabel("Mean toxicity")
    ax1.set_ylabel("Perplexity")
    ax1.set_yscale("log")
    ax1.set_title("Toxicity vs. PPL tradeoff: Pruning vs. Rep-Eng")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Toxicity ratio (steered / baseline)")
    ax2.set_ylabel("PPL ratio (steered / baseline)")
    ax2.set_title("Tox reduction vs. PPL cost: Pruning vs. Rep-Eng")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(1.0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Pruning vs. Representation Engineering: Toxicity–PPL Tradeoff", fontsize=13)
    plt.tight_layout()
    p = output_dir / "prune_vs_repeng_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ── DAA / OSD shared helper ───────────────────────────────────────────────────────────────────

def _plot_generic_pruning_comparison(
    method_results:   dict[str, dict],
    pruning_results:  dict[str, dict],
    output_dir:       Path,
    method_key:       str,
    method_label:     str,
    x_label:          str,
    out_stem:         str,
    out_overlay_stem: str,
) -> None:
    valid_meth = {k: v for k, v in method_results.items()  if v and v.get("pruning_fractions")}
    valid_prun = {k: v for k, v in pruning_results.items() if v and v.get("pruning_fractions")}
    if not valid_meth:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    for i, (label, tp) in enumerate(valid_meth.items()):
        fracs  = tp["pruning_fractions"]
        bt     = tp["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
        bp     = tp["unpruned"]["ppl"]
        bv     = tp["unpruned"].get("val_loss", float("nan"))
        p_tox  = [tp["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in fracs]
        p_ppls = [tp["pruned"][str(f)]["perplexity"] for f in fracs]
        p_vls  = [tp["pruned"][str(f)].get("val_loss", float("nan")) for f in fracs]
        color  = COLORS[i % len(COLORS)]
        x_pct  = [f * 100 for f in fracs]
        ax1.plot([0] + x_pct, [bt] + p_tox,  "o-", label=label, color=color, linewidth=2)
        ax2.plot([0] + x_pct, [bp] + p_ppls, "s-", label=label, color=color, linewidth=2)
        ax3.plot([0] + x_pct, [bv] + p_vls,  "^-", label=label, color=color, linewidth=2)
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    for ax, ylabel, title in [
        (ax1, "Toxicity (mean)", f"Toxicity Reduction via {method_label}"),
        (ax2, "Perplexity",     f"Perplexity Cost of {method_label}"),
        (ax3, "Val Loss",       f"Val Loss Cost of {method_label}"),
    ]:
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{method_label} — All Models", fontsize=13)
    plt.tight_layout()
    p = output_dir / f"{out_stem}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")

    all_labels = sorted(set(list(valid_meth.keys()) + list(valid_prun.keys())))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, label in enumerate(all_labels):
        color = COLORS[i % len(COLORS)]
        if label in valid_prun:
            tp      = valid_prun[label]
            p_fracs = tp["pruning_fractions"]
            bt      = tp["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
            bp      = tp["unpruned"]["ppl"]
            p_tox   = [tp["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in p_fracs]
            p_ppls  = [tp["pruned"][str(f)]["perplexity"] for f in p_fracs]
            ax1.plot([0] + [f * 100 for f in p_fracs], [bt] + p_tox,
                     "o--", color=color, linewidth=2, alpha=0.8, label=f"{label} (neuron)")
            ax2.plot([0] + [f * 100 for f in p_fracs], [bp] + p_ppls,
                     "s--", color=color, linewidth=2, alpha=0.8, label=f"{label} (neuron)")
        if label in valid_meth:
            tp      = valid_meth[label]
            a_fracs = tp["pruning_fractions"]
            bt      = tp["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
            bp      = tp["unpruned"]["ppl"]
            a_tox   = [tp["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in a_fracs]
            a_ppls  = [tp["pruned"][str(f)]["perplexity"] for f in a_fracs]
            ax1.plot([0] + [f * 100 for f in a_fracs], [bt] + a_tox,
                     "o-", color=color, linewidth=2, label=f"{label} ({method_key})")
            ax2.plot([0] + [f * 100 for f in a_fracs], [bp] + a_ppls,
                     "s-", color=color, linewidth=2, label=f"{label} ({method_key})")
    ax2.set_yscale("log")
    for ax, ylabel, title in [
        (ax1, "Toxicity (mean)", f"Neuron Pruning vs. {method_label} — Toxicity"),
        (ax2, "Perplexity",     f"Neuron Pruning vs. {method_label} — PPL"),
    ]:
        ax.set_xlabel("Fraction (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Neuron Pruning vs. {method_label}", fontsize=13)
    plt.tight_layout()
    p = output_dir / f"{out_overlay_stem}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def plot_daa_comparison(
    daa_results:     dict[str, dict],
    pruning_results: dict[str, dict],
    output_dir:      Path,
) -> None:
    """Cross-model comparison for DAA pruning.

    Saves ``daa_comparison.png`` and ``neuron_vs_daa_comparison.png``.
    """
    _plot_generic_pruning_comparison(
        method_results=daa_results,
        pruning_results=pruning_results,
        output_dir=output_dir,
        method_key="DAA",
        method_label="DAA Pruning (Differential Activation)",
        x_label="Projection strength α (%)",
        out_stem="daa_comparison",
        out_overlay_stem="neuron_vs_daa_comparison",
    )


def plot_osd_comparison(
    osd_results:     dict[str, dict],
    pruning_results: dict[str, dict],
    output_dir:      Path,
) -> None:
    """Cross-model comparison for OSD pruning.

    Saves ``osd_comparison.png`` and ``neuron_vs_osd_comparison.png``.
    """
    _plot_generic_pruning_comparison(
        method_results=osd_results,
        pruning_results=pruning_results,
        output_dir=output_dir,
        method_key="OSD",
        method_label="OSD Pruning (Orthogonal Subspace)",
        x_label="PC Fraction Removed (%)",
        out_stem="osd_comparison",
        out_overlay_stem="neuron_vs_osd_comparison",
    )


def save_svd_visualizations(
    svd_sel:  dict,   # layer_idx (str) \u2192 {singular_values, component_scores, ...}
    svd_prun: dict,   # {pruning_fractions, unpruned, pruned}
    label:    str,
    vis_dir:  Path,
) -> None:
    """
    Per-model SVD selectivity and pruning visualizations saved to *vis_dir*.

    Plots
    -----
    1. svd_pruning_curves.png        \u2014 per-model toxicity + PPL vs. fraction
    2. svd_component_selectivity.png \u2014 per-layer sorted selectivity bar charts
                                       with cutoff lines per fraction
    3. svd_singularval_vs_selectivity.png \u2014 scatter (s_k, sel_k) per layer,
                                       pruned components coloured red
    """
    vis_dir.mkdir(parents=True, exist_ok=True)

    svd_sel_int = {int(k): v for k, v in svd_sel.items()}
    n_layers = len(svd_sel_int)
    if n_layers == 0:
        return

    frac_colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#984ea3"]
    fracs = svd_prun.get("pruning_fractions", [])

    # \u2500\u2500 1. SVD pruning curves \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if fracs:
        baseline_tox = svd_prun["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
        pruned_tox   = [svd_prun["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"]
                        for f in fracs]
        baseline_ppl = svd_prun["unpruned"]["ppl"]
        ppls         = [svd_prun["pruned"][str(f)]["perplexity"] for f in fracs]
        norm_tox     = [v / max(baseline_tox, 1e-8) for v in pruned_tox]
        x_pct        = [f * 100 for f in fracs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot([0] + x_pct, [1.0] + norm_tox, "o-", color="#d62728", linewidth=2)
        ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_xlabel("SVD components pruned (%)")
        ax1.set_ylabel("Toxicity (relative to unpruned)")
        ax1.set_title("Toxicity reduction vs. SVD pruning")
        ax1.grid(True, alpha=0.3)

        ax2.plot([0] + x_pct, [baseline_ppl] + ppls, "s-", color="#1f77b4", linewidth=2)
        ax2.set_yscale("log")
        ax2.set_xlabel("SVD components pruned (%)")
        ax2.set_ylabel("Perplexity")
        ax2.set_title("PPL cost of SVD pruning")
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"{label} \u2014 SVD-direction toxicity pruning sweep", fontsize=13)
        plt.tight_layout()
        p = vis_dir / "svd_pruning_curves.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"      \u2192 {p}")

    # \u2500\u2500 2. Per-layer component selectivity spectra \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ncols = 4
    nrows = (n_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for i in range(n_layers):
        ax     = axes_flat[i]
        info   = svd_sel_int[i]
        scores = np.array(info["component_scores"])
        svals  = np.array(info["singular_values"])
        order  = np.argsort(scores)[::-1]
        sorted_scores = scores[order]
        sorted_svals  = svals[order]
        sv_norm = sorted_svals / (sorted_svals.max() + 1e-12)
        x_idx = np.arange(len(sorted_scores))
        ax.bar(x_idx, sorted_scores, color="#888888", alpha=0.5, linewidth=0)
        ax.scatter(x_idx, sorted_scores, c=sv_norm, cmap="viridis",
                   s=8, zorder=3, alpha=0.8)
        n_comp = len(scores)
        for fi, frac in enumerate(fracs):
            cutoff = max(1, int(n_comp * frac)) - 1
            c = frac_colors[fi % len(frac_colors)]
            ax.axvline(cutoff, color=c, linestyle="--", linewidth=1.2,
                       label=f"{frac:.0%}" if i == 0 else None)
        er = info.get("effective_rank", float("nan"))
        ax.set_title(f"L{i}  (eff.rank={er:.1f})", fontsize=9)
        ax.set_xlabel("Component rank (\u2192 low sel.)", fontsize=7)
        ax.set_ylabel("Selectivity |v\u00b7t\u0302|", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(n_layers, len(axes_flat)):
        axes_flat[j].axis("off")
    if fracs:
        axes_flat[0].legend(title="Prune frac", fontsize=7, title_fontsize=7)
    fig.suptitle(f"{label} \u2014 SVD component selectivity (sorted, cutoffs shown)", fontsize=12)
    plt.tight_layout()
    p = vis_dir / "svd_component_selectivity.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"      \u2192 {p}")

    # \u2500\u2500 3. Singular value vs. selectivity scatter \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    max_frac = max(fracs) if fracs else 0.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for i in range(n_layers):
        ax   = axes_flat[i]
        info = svd_sel_int[i]
        scores = np.array(info["component_scores"])
        svals  = np.array(info["singular_values"])
        n_comp = len(scores)
        n_prune = max(1, int(n_comp * max_frac)) if max_frac > 0 else 0
        pruned_mask = np.zeros(n_comp, dtype=bool)
        if n_prune > 0:
            pruned_mask[np.argsort(scores)[-n_prune:]] = True
        ax.scatter(scores[~pruned_mask], svals[~pruned_mask],
                   s=12, color="#aaaaaa", alpha=0.7, label="kept")
        if pruned_mask.any():
            ax.scatter(scores[pruned_mask], svals[pruned_mask],
                       s=20, color="#d62728", alpha=0.9,
                       label=f"pruned @{max_frac:.0%}")
        er = info.get("effective_rank", float("nan"))
        ax.set_title(f"L{i}  (eff.rank={er:.1f})", fontsize=9)
        ax.set_xlabel("Selectivity |v\u00b7t\u0302|", fontsize=7)
        ax.set_ylabel("Singular value", fontsize=7)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n_layers, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle(
        f"{label} \u2014 Singular value vs. selectivity  "
        f"(red = pruned at {max_frac:.0%})",
        fontsize=12,
    )
    plt.tight_layout()
    p = vis_dir / "svd_singularval_vs_selectivity.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"      \u2192 {p}")


def save_amplification_visualizations(
    amp_result:        dict,
    pruning_result:    dict | None,
    t_stats_per_layer: dict[int, np.ndarray] | None,
    label:             str,
    vis_dir:           Path,
) -> None:
    """
    Per-model amplification visualizations saved to *vis_dir*.

    Plots
    -----
    1. amplification_curves.png              — toxicity + PPL vs. fraction amplified,
                                               overlaid with pruning curves for comparison
    2. amplified_neurons_cortical_{f}pct.png — cortical sheet with amplified
                                               neurons shown in green (vs yellow for pruned)
    """
    vis_dir.mkdir(parents=True, exist_ok=True)

    fracs      = amp_result.get("amp_fracs", [])
    amp_factor = amp_result.get("amp_factor", 1.0)
    if not fracs:
        return

    base_tox = amp_result["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
    base_ppl = amp_result["unpruned"]["ppl"]
    amp_tox  = [amp_result["amplified"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in fracs]
    amp_ppls = [amp_result["amplified"][str(f)]["perplexity"] for f in fracs]
    norm_amp = [v / max(base_tox, 1e-8) for v in amp_tox]
    x_pct    = [f * 100 for f in fracs]

    # ── 1. Curves (pruning dashed for comparison) ─────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if pruning_result and pruning_result.get("pruning_fractions"):
        p_fracs = pruning_result["pruning_fractions"]
        p_bt    = pruning_result["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
        p_bp    = pruning_result["unpruned"]["ppl"]
        p_tox   = [pruning_result["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in p_fracs]
        p_ppls  = [pruning_result["pruned"][str(f)]["perplexity"] for f in p_fracs]
        ax1.plot([0] + [f * 100 for f in p_fracs], [1.0] + [v / max(p_bt, 1e-8) for v in p_tox],
                 "o--", color="#d62728", linewidth=2, alpha=0.7, label="pruned (zero)")
        ax2.plot([0] + [f * 100 for f in p_fracs], [p_bp] + p_ppls,
                 "s--", color="#1f77b4", linewidth=2, alpha=0.7, label="pruned (zero)")

    ax1.plot([0] + x_pct, [1.0] + norm_amp, "o-", color="#ff7f0e", linewidth=2,
             label=f"amplified (×{amp_factor})")
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Neurons selected (%)")
    ax1.set_ylabel("Toxicity (relative to unpruned)")
    ax1.set_title("Toxicity: pruning vs. amplification")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot([0] + x_pct, [base_ppl] + amp_ppls, "s-", color="#2ca02c", linewidth=2,
             label=f"amplified (×{amp_factor})")
    ax2.set_yscale("log")
    ax2.set_xlabel("Neurons selected (%)")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("PPL: pruning vs. amplification")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{label} — Toxic-neuron amplification sweep (×{amp_factor})", fontsize=13)
    plt.tight_layout()
    p = vis_dir / "amplification_curves.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"      \u2192 {p}")

    # ── 2. Cortical-sheet overlays with green = amplified neurons ───────────────────────────────
    if t_stats_per_layer is None:
        return
    n_layers = len(t_stats_per_layer)
    if n_layers == 0:
        return

    n_neurons = len(t_stats_per_layer[0])
    H, W = _cortical_sheet_dims(n_neurons)
    t_all_arr = np.array([t_stats_per_layer[i] for i in range(n_layers)])
    vmax = float(np.percentile(np.abs(t_all_arr), 97))

    for frac in fracs:
        n_amp   = max(1, int(n_neurons * frac))
        ncols_m = 4
        nrows_m = (n_layers + ncols_m - 1) // ncols_m
        fig, axes_m = plt.subplots(nrows_m, ncols_m, figsize=(4 * ncols_m, 4 * nrows_m))
        axes_flat_m = list(axes_m.flat) if hasattr(axes_m, "flat") else [axes_m]

        for i in range(n_layers):
            ax = axes_flat_m[i]
            t  = t_stats_per_layer[i]
            ax.imshow(t.reshape(H, W).astype(float), cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
            amp_idx = np.argsort(t)[-n_amp:]
            mask    = np.zeros(n_neurons, dtype=float)
            mask[amp_idx] = 1.0
            overlay = np.zeros((H, W, 4), dtype=float)
            overlay[..., 1] = 0.85   # green channel
            overlay[..., 3] = mask.reshape(H, W) * 0.85
            ax.imshow(overlay, aspect="auto", interpolation="nearest")
            ax.set_title(f"L{i}  ({n_amp / n_neurons * 100:.1f}% amplified)", fontsize=9)
            ax.axis("off")

        for j in range(n_layers, len(axes_flat_m)):
            axes_flat_m[j].axis("off")
        fig.suptitle(
            f"{label} — Amplified neurons on cortical sheet  |  "
            f"fraction={frac:.0%}  ({n_amp}/{n_neurons} per layer)  ×{amp_factor}\n"
            f"Background: t-stat (red=toxic)   Overlay: amplified neurons (green)",
            fontsize=11,
        )
        plt.tight_layout()
        frac_str = f"{int(frac * 100):02d}"
        p = vis_dir / f"amplified_neurons_cortical_{frac_str}pct.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"      \u2192 {p}")


def save_global_selectivity_visualizations(
    t_stats_per_layer: dict[int, np.ndarray],
    pruning_result:    dict,
    label:             str,
    vis_dir:           Path,
) -> None:
    """
    Visualizations unique to the cross-layer global pruning strategy.
    Saved alongside the per-layer files in *vis_dir* with a ``global_`` prefix.

    Plots
    -----
    1. global_pruning_curves.png         — toxicity + PPL vs. global fraction
    2. global_pruned_per_layer_dist.png  — neurons pruned per layer per fraction
    3. global_pruned_neurons_cortical_Xpct.png — cortical sheet overlays (orange)
    4. global_layer_frac_pruned_Xpct.png — per fraction: % of each layer pruned
    """
    vis_dir.mkdir(parents=True, exist_ok=True)
    n_layers = len(t_stats_per_layer)
    if n_layers == 0:
        return

    fracs = pruning_result.get("global_pruning_fractions", [])
    if not fracs:
        return

    layer_indices  = sorted(t_stats_per_layer.keys())
    all_t_flat     = np.concatenate([t_stats_per_layer[i] for i in layer_indices])
    sizes          = np.array([len(t_stats_per_layer[i]) for i in layer_indices])
    boundaries     = np.concatenate([[0], np.cumsum(sizes)])
    total_neurons  = int(boundaries[-1])

    def _global_prune_by_layer(frac: float) -> dict[int, np.ndarray]:
        n_prune  = max(1, int(total_neurons * frac))
        top_flat = np.argpartition(all_t_flat, -n_prune)[-n_prune:]
        by_layer: dict[int, list[int]] = {i: [] for i in layer_indices}
        for flat_idx in top_flat.tolist():
            li  = int(np.searchsorted(boundaries[1:], flat_idx, side="right"))
            by_layer[layer_indices[li]].append(int(flat_idx) - int(boundaries[li]))
        return {i: np.array(v, dtype=int) for i, v in by_layer.items()}

    # ── 1. Global pruning curves ───────────────────────────────────────────────────
    baseline_tox = pruning_result["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
    pruned_tox   = [pruning_result["pruned"][str(f)]["toxicity_scores"]["toxicity"]["mean"] for f in fracs]
    baseline_ppl = pruning_result["unpruned"]["ppl"]
    ppls         = [pruning_result["pruned"][str(f)]["perplexity"] for f in fracs]
    baseline_vl  = pruning_result["unpruned"]["val_loss"]
    vls          = [pruning_result["pruned"][str(f)]["val_loss"] for f in fracs]
    x_pct        = [f * 100 for f in fracs]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.plot([0] + x_pct, [baseline_tox] + pruned_tox, "o-", color="#d62728", linewidth=2)
    ax1.set_xlabel("Neurons pruned globally (%)")
    ax1.set_ylabel("Toxicity (mean)")
    ax1.set_title("Toxicity reduction vs. global pruning")
    ax1.grid(True, alpha=0.3)

    ax2.plot([0] + x_pct, [baseline_ppl] + ppls, "s-", color="#1f77b4", linewidth=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("Neurons pruned globally (%)")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("PPL cost of global pruning")
    ax2.grid(True, alpha=0.3)

    ax3.plot([0] + x_pct, [baseline_vl] + vls, "^-", color="#2ca02c", linewidth=2)
    ax3.set_yscale("log")
    ax3.set_xlabel("Neurons pruned globally (%)")
    ax3.set_ylabel("Val Loss")
    ax3.set_title("Val loss cost of global pruning")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f"{label} — Global toxicity pruning sweep", fontsize=13)
    plt.tight_layout()
    p = vis_dir / "global_pruning_curves.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"      → {p}")

    # ── 2. Per-layer pruned-neuron distribution ─────────────────────────────────────────
    frac_colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#984ea3"]
    x_pos = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(max(10, n_layers * 0.8), 5))
    bar_width = 0.8 / max(len(fracs), 1)
    for fi, frac in enumerate(fracs):
        by_layer = _global_prune_by_layer(frac)
        counts   = [len(by_layer.get(i, np.array([]))) for i in range(n_layers)]
        offset   = (fi - len(fracs) / 2 + 0.5) * bar_width
        ax.bar(x_pos + offset, counts, bar_width * 0.9,
               label=f"{frac:.0%}", color=frac_colors[fi % len(frac_colors)], alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Neurons pruned")
    ax.set_title(f"{label} — Neurons per layer pruned at each global fraction")
    ax.legend(title="Global frac")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p = vis_dir / "global_pruned_per_layer_dist.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"      → {p}")

    # ── 3. Cortical-sheet overlays (yellow = globally pruned) ─────────────────────────────
    n_neurons = len(t_stats_per_layer[0])
    H, W      = _cortical_sheet_dims(n_neurons)
    t_all_arr = np.array([t_stats_per_layer[i] for i in range(n_layers)])
    vmax      = float(np.percentile(np.abs(t_all_arr), 97))

    for frac in fracs:
        by_layer = _global_prune_by_layer(frac)
        n_total_pruned = sum(len(v) for v in by_layer.values())

        ncols_m = 4
        nrows_m = (n_layers + ncols_m - 1) // ncols_m
        fig, axes_m = plt.subplots(nrows_m, ncols_m, figsize=(4 * ncols_m, 4 * nrows_m))
        axes_flat_m = list(axes_m.flat) if hasattr(axes_m, "flat") else [axes_m]

        for i in range(n_layers):
            ax = axes_flat_m[i]
            t  = t_stats_per_layer[i]
            ax.imshow(t.reshape(H, W).astype(float), cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
            pruned_idx = by_layer.get(i, np.array([], dtype=int))
            mask = np.zeros(n_neurons, dtype=float)
            if len(pruned_idx) > 0:
                mask[pruned_idx] = 1.0
            overlay = np.zeros((H, W, 4), dtype=float)
            overlay[..., 0] = 1.0
            overlay[..., 1] = 0.95
            overlay[..., 2] = 0.0
            overlay[..., 3] = mask.reshape(H, W) * 0.85
            ax.imshow(overlay, aspect="auto", interpolation="nearest")
            n_this = len(pruned_idx)
            ax.set_title(f"L{i}  ({n_this} pruned)", fontsize=9)
            ax.axis("off")

        for j in range(n_layers, len(axes_flat_m)):
            axes_flat_m[j].axis("off")
        fig.suptitle(
            f"{label} — Globally pruned neurons on cortical sheet  |  "
            f"global fraction={frac:.0%}  ({n_total_pruned} total across {n_layers} layers)\n"
            f"Background: t-stat (red=toxic)   Overlay: pruned neurons (yellow)",
            fontsize=11,
        )
        plt.tight_layout()
        frac_str = f"{int(frac * 100):02d}"
        p = vis_dir / f"global_pruned_neurons_cortical_{frac_str}pct.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"      → {p}")

    # ── 4. Per-fraction: fraction of each layer that was pruned ───────────────
    layer_sizes = np.array([len(t_stats_per_layer[i]) for i in range(n_layers)])

    for frac in fracs:
        by_layer   = _global_prune_by_layer(frac)
        counts     = np.array([len(by_layer.get(i, np.array([]))) for i in range(n_layers)],
                              dtype=float)
        pct_pruned = counts / layer_sizes * 100.0

        fig, ax = plt.subplots(figsize=(6, max(4, n_layers * 0.45)))
        bars = ax.barh(
            [f"L{i}" for i in range(n_layers)],
            pct_pruned,
            color="#ff7f0e", edgecolor="white", linewidth=0.5, alpha=0.9,
        )
        ax.axvline(frac * 100, color="#d62728", linestyle="--", linewidth=1.5,
                   label=f"Global budget ({frac:.0%})")
        for bar, cnt in zip(bars, counts.astype(int)):
            if cnt > 0:
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                        f"{cnt}", va="center", ha="left", fontsize=8)
        ax.set_xlabel("Neurons pruned (% of layer)")
        ax.set_title(
            f"{label}\nPer-layer pruning share — global fraction {frac:.0%}\n"
            f"({int(counts.sum())} total neurons pruned across {n_layers} layers)"
        )
        ax.legend(fontsize=9)
        ax.set_xlim(0, max(pct_pruned.max() * 1.15, frac * 100 * 1.3))
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()
        plt.tight_layout()
        frac_str = f"{int(frac * 100):02d}"
        p = vis_dir / f"global_layer_frac_pruned_{frac_str}pct.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"      → {p}")


# ── JSON discovery and loading ─────────────────────────────────────────────────

def discover_pruning_jsons(output_dir: Path) -> dict[str, dict]:
    """
    Find all pruning_tau*.json (and companion t_stats_tau*.json) files in
    output_dir and return a dict keyed by the model label recovered from
    results.json (or synthesised from the tau value).

    t_stats_per_layer (if present in t_stats_tau{tau}.json) is merged back
    into the pruning dict so the replot script can regenerate cortical sheet
    and t-stat distribution plots without a GPU run.
    """
    results_path = output_dir / "results.json"
    results      = _load_json(results_path) or {}

    # Build tau → label mapping from results.json keys (e.g. "tau=0.5")
    tau_to_label: dict[str, str] = {}
    for label in results:
        # labels look like "tau=0.0 (baseline)" or "tau=0.5"
        for part in label.split():
            if part.startswith("tau="):
                raw_tau = part.split("=")[1]
                tau_to_label[raw_tau] = label
                break

    pruning: dict[str, dict] = {}
    for p in sorted(output_dir.glob("pruning_tau*.json")):
        # filename: pruning_tau0_5.json  → tau string "0.5"
        stem    = p.stem.removeprefix("pruning_tau")   # e.g. "0_5" or "0_0"
        tau_str = stem.replace("_", ".")               # "0.5" or "0.0"
        label   = tau_to_label.get(tau_str, f"tau={tau_str}")
        data    = _load_json(p)
        if not data:
            continue

        # Try to load companion t_stats file (saved by eval script)
        t_stats_path = output_dir / f"t_stats_tau{stem}.json"
        if t_stats_path.is_file():
            t_stats = _load_json(t_stats_path)
            if t_stats:
                data["t_stats_per_layer"] = t_stats
                print(f"  Loaded pruning + t_stats for {label} ← {p.name} + {t_stats_path.name}")
            else:
                print(f"  Loaded pruning JSON for {label} ← {p.name} (no t_stats)")
        else:
            print(f"  Loaded pruning JSON for {label} ← {p.name} (no t_stats_tau{stem}.json)")

        pruning[label] = data

    return pruning


def discover_svd_jsons(
    output_dir:  Path,
    tau_to_label: dict[str, str],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Discover svd_selectivity_tau*.json and svd_pruning_tau*.json files.
    Returns (svd_selectivity_all, svd_pruning_results) dicts keyed by label.
    """
    svd_selectivity_all: dict[str, dict] = {}
    svd_pruning_results: dict[str, dict] = {}

    for p in sorted(output_dir.glob("svd_selectivity_tau*.json")):
        stem    = p.stem.removeprefix("svd_selectivity_tau")
        tau_str = stem.replace("_", ".")
        label   = tau_to_label.get(tau_str, f"tau={tau_str}")
        data    = _load_json(p)
        if data:
            svd_selectivity_all[label] = data
            print(f"  Loaded SVD selectivity for {label} ← {p.name}")

    for p in sorted(output_dir.glob("svd_pruning_tau*.json")):
        stem    = p.stem.removeprefix("svd_pruning_tau")
        tau_str = stem.replace("_", ".")
        label   = tau_to_label.get(tau_str, f"tau={tau_str}")
        data    = _load_json(p)
        if data:
            svd_pruning_results[label] = data
            print(f"  Loaded SVD pruning   for {label} ← {p.name}")

    return svd_selectivity_all, svd_pruning_results


def discover_amplification_jsons(
    output_dir:   Path,
    tau_to_label: dict[str, str],
) -> dict[str, dict]:
    """
    Discover amplification_tau*.json files.
    Returns a dict keyed by model label.
    """
    amp_results: dict[str, dict] = {}
    for p in sorted(output_dir.glob("amplification_tau*.json")):
        stem    = p.stem.removeprefix("amplification_tau")
        tau_str = stem.replace("_", ".")
        label   = tau_to_label.get(tau_str, f"tau={tau_str}")
        data    = _load_json(p)
        if data:
            amp_results[label] = data
            print(f"  Loaded amplification for {label} ← {p.name}")
    return amp_results


def discover_global_pruning_jsons(
    output_dir:   Path,
    tau_to_label: dict[str, str],
) -> dict[str, dict]:
    """
    Discover global_pruning_tau*.json files.
    Returns a dict keyed by model label.
    """
    global_results: dict[str, dict] = {}
    for p in sorted(output_dir.glob("global_pruning_tau*.json")):
        stem    = p.stem.removeprefix("global_pruning_tau")
        tau_str = stem.replace("_", ".")
        label   = tau_to_label.get(tau_str, f"tau={tau_str}")
        data    = _load_json(p)
        if data:
            global_results[label] = data
            print(f"  Loaded global pruning for {label} ← {p.name}")
    return global_results


def discover_attenuation_jsons(
    output_dir:   Path,
    tau_to_label: dict[str, str],
) -> dict[str, dict]:
    """Discover attenuation_tau*.json files. Returns dict keyed by model label."""
    att_results: dict[str, dict] = {}
    for p in sorted(output_dir.glob("attenuation_tau*.json")):
        stem    = p.stem.removeprefix("attenuation_tau")
        tau_str = stem.replace("_", ".")
        label   = tau_to_label.get(tau_str, f"tau={tau_str}")
        data    = _load_json(p)
        if data:
            att_results[label] = data
            print(f"  Loaded attenuation for {label} ← {p.name}")
    return att_results


def discover_pca_pruning_jsons(
    output_dir:   Path,
    tau_to_label: dict[str, str],
) -> dict[str, dict]:
    """Discover pca_pruning_tau*.json files. Returns dict keyed by model label."""
    pca_results: dict[str, dict] = {}
    for p in sorted(output_dir.glob("pca_pruning_tau*.json")):
        stem    = p.stem.removeprefix("pca_pruning_tau")
        tau_str = stem.replace("_", ".")
        label   = tau_to_label.get(tau_str, f"tau={tau_str}")
        data    = _load_json(p)
        if data:
            pca_results[label] = data
            print(f"  Loaded PCA pruning for {label} ← {p.name}")
    return pca_results


def discover_daa_pruning_jsons(
    output_dir:   Path,
    tau_to_label: dict[str, str],
) -> dict[str, dict]:
    """Discover daa_pruning_tau*.json files. Returns dict keyed by model label."""
    daa_results: dict[str, dict] = {}
    for p in sorted(output_dir.glob("daa_pruning_tau*.json")):
        stem    = p.stem.removeprefix("daa_pruning_tau")
        tau_str = stem.replace("_", ".")
        label   = tau_to_label.get(tau_str, f"tau={tau_str}")
        data    = _load_json(p)
        if data:
            daa_results[label] = data
            print(f"  Loaded DAA pruning for {label} \u2190 {p.name}")
    return daa_results


def discover_osd_pruning_jsons(
    output_dir:   Path,
    tau_to_label: dict[str, str],
) -> dict[str, dict]:
    """Discover osd_pruning_tau*.json files. Returns dict keyed by model label."""
    osd_results: dict[str, dict] = {}
    for p in sorted(output_dir.glob("osd_pruning_tau*.json")):
        stem    = p.stem.removeprefix("osd_pruning_tau")
        tau_str = stem.replace("_", ".")
        label   = tau_to_label.get(tau_str, f"tau={tau_str}")
        data    = _load_json(p)
        if data:
            osd_results[label] = data
            print(f"  Loaded OSD pruning for {label} \u2190 {p.name}")
    return osd_results


def discover_repeng_jsons(
    output_dir:   Path,
    tau_to_label: dict[str, str],
) -> dict[str, dict]:
    """Discover repeng_tau*.json files. Returns dict keyed by model label."""
    repeng_results: dict[str, dict] = {}
    for p in sorted(output_dir.glob("repeng_tau*.json")):
        stem    = p.stem.removeprefix("repeng_tau")
        tau_str = stem.replace("_", ".")
        label   = tau_to_label.get(tau_str, f"tau={tau_str}")
        data    = _load_json(p)
        if data:
            repeng_results[label] = data
            print(f"  Loaded rep-eng for {label} ← {p.name}")
    return repeng_results


# ── results.json synthesis from secondary JSON files ────────────────────────────

def synthesize_results_json(output_dir: Path) -> dict | None:
    """
    Reconstruct a results.json-compatible dict when results.json is missing.

    Every pruning_tau*.json and amplification_tau*.json file contains an
    'unpruned' block with the full toxicity_scores and ppl for the baseline
    (un-modified) model — the same values that would have been written to
    results.json.  We use those to recreate the summary dict.

    Label format is inferred from the output directory name so it matches
    what the eval script would have written:
      toxicity_nanogpt                 → "tau=0.0 (baseline)", "tau=0.5" …
      toxicity_nanogpt_quantized_fp16  → "tau=0.0 (baseline) [fp16]" …

    The synthesized dict is written to results.json so subsequent replot
    invocations don't need to synthesize again.

    Returns the synthesized dict, or None if no source files are found.
    """
    # Detect quantization suffix from directory name
    dir_name  = output_dir.name
    quant_tag = ""
    for qt in ("fp16", "bf16", "int8", "int4"):
        if dir_name.endswith(f"_{qt}"):
            quant_tag = f" [{qt}]"
            break

    # Collect (tau_float, tau_str_as_in_filename, unpruned_block) triples.
    # Prefer pruning JSONs; fall back to amplification JSONs.
    sources: dict[str, dict] = {}  # tau_str → unpruned block

    for pattern, prefix in [
        ("pruning_tau*.json",       "pruning_tau"),
        ("amplification_tau*.json", "amplification_tau"),
    ]:
        for p in sorted(output_dir.glob(pattern)):
            stem    = p.stem.removeprefix(prefix)   # e.g. "0_5" or "0_0"
            tau_str = stem.replace("_", ".")        # "0.5" or "0.0"
            if tau_str in sources:                  # pruning already registered
                continue
            data = _load_json(p)
            if data and "unpruned" in data:
                sources[tau_str] = data["unpruned"]

    if not sources:
        return None

    BASELINE_TAU = "0.0"
    results: dict[str, dict] = {}

    for tau_str, unpruned in sorted(sources.items(),
                                    key=lambda kv: float(kv[0])):
        baseline_marker = " (baseline)" if tau_str == BASELINE_TAU else ""
        label = f"tau={tau_str}{baseline_marker}{quant_tag}"
        results[label] = {
            "quantization":           quant_tag.strip(" []") or "fp32",
            "n_prompts":              0,   # not stored in secondary JSONs
            "n_completions":          0,
            "toxicity_scores":        unpruned["toxicity_scores"],
            "perplexity":             unpruned.get("ppl", float("nan")),
            "per_completion_toxicity": [],  # not available
        }

    # Write synthesized results.json so future replots don't need to synthesize.
    out_path = output_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(
            # strip internal-only keys before saving
            {
                k: {ik: iv for ik, iv in v.items() if ik != "per_completion_toxicity"}
                for k, v in results.items()
            },
            f, indent=2,
        )
    print(f"  Synthesized and saved results.json → {out_path}")
    print(f"  NOTE: n_prompts / n_completions unknown; per-prompt heatmap will be skipped.")
    return results


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-generate all toxicity plots from saved JSON results"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory containing results.json and pruning_tau*.json "
             "(default: outputs/toxicity_nanogpt relative to repo root)",
    )
    parser.add_argument(
        "--no_selectivity", action="store_true",
        help="Skip per-model selectivity subdirectory plots",
    )
    parser.add_argument(
        "--no_svd", action="store_true",
        help="Skip SVD effective-rank and SVD pruning comparison plots",
    )
    parser.add_argument(
        "--no_amplification", action="store_true",
        help="Skip toxic-neuron amplification plots",
    )
    parser.add_argument(
        "--no_attenuation", action="store_true",
        help="Skip toxic-neuron attenuation plots",
    )
    parser.add_argument(
        "--no_pca_pruning", action="store_true",
        help="Skip PCA pruning plots",
    )
    parser.add_argument(
        "--no_daa_pruning", action="store_true",
        help="Skip DAA pruning plots",
    )
    parser.add_argument(
        "--no_osd_pruning", action="store_true",
        help="Skip OSD pruning plots",
    )
    parser.add_argument(
        "--no_repeng", action="store_true",
        help="Skip representation-engineering steering plots",
    )
    parser.add_argument(
        "--no_global_pruning", action="store_true",
        help="Skip cross-layer global neuron pruning plots",
    )
    return parser.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "outputs" / "toxicity_nanogpt"

    if not output_dir.is_dir():
        raise SystemExit(f"Output directory not found: {output_dir}")

    print(f"Output directory: {output_dir}")
    print()

    # ── Load results.json (synthesize from pruning/amp JSONs if missing) ─────
    results_path = output_dir / "results.json"
    results      = _load_json(results_path)
    if not results:
        print("results.json not found — attempting to synthesize from available JSON files…")
        results = synthesize_results_json(output_dir)
        if not results:
            raise SystemExit(
                f"Cannot generate results.json: no pruning_tau*.json or "
                f"amplification_tau*.json files found in {output_dir}"
            )
    print(f"Loaded results.json — {len(results)} model(s): {list(results.keys())}")
    print()

    # ── Load pruning JSONs ─────────────────────────────────────────────────
    print("Discovering pruning JSON files…")
    pruning_results = discover_pruning_jsons(output_dir)
    print(f"  Found {len(pruning_results)} pruning result(s).")

    # Build tau→label map for SVD JSON discovery (reuse from discover_pruning_jsons)
    tau_to_label: dict[str, str] = {}
    for label in results:
        for part in label.split():
            if part.startswith("tau="):
                tau_to_label[part.split("=")[1]] = label
                break

    svd_selectivity_all:    dict[str, dict] = {}
    svd_pruning_results:    dict[str, dict] = {}
    amp_results:            dict[str, dict] = {}
    att_results:            dict[str, dict] = {}
    pca_pruning_results:    dict[str, dict] = {}
    daa_pruning_results:    dict[str, dict] = {}
    osd_pruning_results:    dict[str, dict] = {}
    repeng_results:         dict[str, dict] = {}
    global_pruning_results: dict[str, dict] = {}
    if not args.no_svd:
        print("Discovering SVD JSON files…")
        svd_selectivity_all, svd_pruning_results = discover_svd_jsons(output_dir, tau_to_label)
        print(f"  Found {len(svd_selectivity_all)} SVD selectivity file(s), "
              f"{len(svd_pruning_results)} SVD pruning file(s).")
    if not args.no_amplification:
        print("Discovering amplification JSON files…")
        amp_results = discover_amplification_jsons(output_dir, tau_to_label)
        print(f"  Found {len(amp_results)} amplification file(s).")
    if not args.no_attenuation:
        print("Discovering attenuation JSON files…")
        att_results = discover_attenuation_jsons(output_dir, tau_to_label)
        print(f"  Found {len(att_results)} attenuation file(s).")
    if not args.no_pca_pruning:
        print("Discovering PCA pruning JSON files…")
        pca_pruning_results = discover_pca_pruning_jsons(output_dir, tau_to_label)
        print(f"  Found {len(pca_pruning_results)} PCA pruning file(s).")
    if not args.no_daa_pruning:
        print("Discovering DAA pruning JSON files…")
        daa_pruning_results = discover_daa_pruning_jsons(output_dir, tau_to_label)
        print(f"  Found {len(daa_pruning_results)} DAA pruning file(s).")
    if not args.no_osd_pruning:
        print("Discovering OSD pruning JSON files…")
        osd_pruning_results = discover_osd_pruning_jsons(output_dir, tau_to_label)
        print(f"  Found {len(osd_pruning_results)} OSD pruning file(s).")
    if not args.no_repeng:
        print("Discovering rep-eng JSON files…")
        repeng_results = discover_repeng_jsons(output_dir, tau_to_label)
        print(f"  Found {len(repeng_results)} rep-eng file(s).")
    if not args.no_global_pruning:
        print("Discovering global pruning JSON files…")
        global_pruning_results = discover_global_pruning_jsons(output_dir, tau_to_label)
        print(f"  Found {len(global_pruning_results)} global pruning file(s).")
    print()

    # ── Cross-model comparison plots ───────────────────────────────────────
    print("Plotting cross-model comparisons…")
    plot_comparison(results, output_dir)

    if pruning_results:
        plot_pruning_comparison(pruning_results, output_dir)
    if global_pruning_results:
        plot_global_pruning_comparison(global_pruning_results, output_dir)
    # ── Amplification comparison ──────────────────────────────────────────
    if not args.no_amplification and amp_results:
        print("Plotting amplification comparison…")
        plot_amplification_comparison(amp_results, pruning_results, output_dir)
    # ── Attenuation comparison ────────────────────────────────────────────
    if not args.no_attenuation and att_results:
        print("Plotting attenuation comparison…")
        plot_attenuation_comparison(att_results, pruning_results, output_dir)
    # ── PCA pruning comparison ────────────────────────────────────────────
    if not args.no_pca_pruning and pca_pruning_results:
        print("Plotting PCA pruning comparison…")
        plot_pca_pruning_comparison(pca_pruning_results, pruning_results, output_dir)
    # ── DAA pruning comparison ─────────────────────────────────────────────
    if not args.no_daa_pruning and daa_pruning_results:
        print("Plotting DAA pruning comparison…")
        plot_daa_comparison(daa_pruning_results, pruning_results, output_dir)
    # ── OSD pruning comparison ─────────────────────────────────────────────
    if not args.no_osd_pruning and osd_pruning_results:
        print("Plotting OSD pruning comparison…")
        plot_osd_comparison(osd_pruning_results, pruning_results, output_dir)
    # ── Rep-eng comparison ────────────────────────────────────────────────
    if not args.no_repeng and repeng_results:
        print("Plotting rep-eng comparison…")
        plot_repeng_comparison(repeng_results, pruning_results, output_dir)
    # ── Effective rank + SVD plots ─────────────────────────────────────
    if not args.no_svd:
        if svd_selectivity_all:
            print("Plotting effective rank and SVD spectra…")
            plot_effective_rank(svd_selectivity_all, output_dir)
        if svd_pruning_results and pruning_results:
            print("Plotting SVD vs. neuron pruning comparison…")
            plot_svd_pruning_comparison(svd_pruning_results, pruning_results, output_dir)
        if svd_pruning_results:
            plot_svd_cross_model_comparison(svd_pruning_results, output_dir)
    # ── Per-model selectivity plots ────────────────────────────────────────
    if not args.no_selectivity and pruning_results:
        sel_dir = output_dir / "selectivity"
        print()
        print(f"Plotting per-model selectivity visualizations → {sel_dir}")
        for label, tp in pruning_results.items():
            raw_t = tp.get("t_stats_per_layer")
            if not raw_t:
                print(f"  [skip] no t_stats_per_layer in pruning JSON for {label}")
                continue
            t_stats_np = {int(k): np.array(v) for k, v in raw_t.items()}
            safe_label = label.replace(" ", "_").replace("=", "")
            vis_dir    = sel_dir / safe_label
            print(f"  {label}  →  {vis_dir.relative_to(output_dir)}/")
            save_selectivity_visualizations(
                t_stats_per_layer=t_stats_np,
                global_stats=tp.get("neuron_stats", {}),
                pruning_result=tp,
                label=label,
                vis_dir=vis_dir,
            )

    # ── Per-model SVD visualizations ────────────────────────────────────────
    if not args.no_svd and not args.no_selectivity:
        sel_dir = output_dir / "selectivity"
        labels_with_both = set(svd_selectivity_all) & set(svd_pruning_results)
        if labels_with_both:
            print()
            print(f"Plotting per-model SVD visualizations → {sel_dir}")
        for label in sorted(labels_with_both):
            safe_label = label.replace(" ", "_").replace("=", "")
            vis_dir    = sel_dir / safe_label
            print(f"  {label}  →  {vis_dir.relative_to(output_dir)}/")
            save_svd_visualizations(
                svd_sel=svd_selectivity_all[label],
                svd_prun=svd_pruning_results[label],
                label=label,
                vis_dir=vis_dir,
            )

    # ── Per-model amplification visualizations ────────────────────────────────
    if not args.no_amplification and not args.no_selectivity and amp_results:
        sel_dir = output_dir / "selectivity"
        print()
        print(f"Plotting per-model amplification visualizations → {sel_dir}")
        for label, ta in amp_results.items():
            safe_label     = label.replace(" ", "_").replace("=", "")
            vis_dir        = sel_dir / safe_label
            pruning_result = pruning_results.get(label)
            raw_t          = (pruning_result or {}).get("t_stats_per_layer")
            t_stats_np     = ({int(k): np.array(v) for k, v in raw_t.items()}
                              if raw_t else None)
            print(f"  {label}  →  {vis_dir.relative_to(output_dir)}/")
            save_amplification_visualizations(
                amp_result=ta,
                pruning_result=pruning_result,
                t_stats_per_layer=t_stats_np,
                label=label,
                vis_dir=vis_dir,
            )

    # ── Per-model global-pruning visualizations ─────────────────────────────
    if not args.no_global_pruning and not args.no_selectivity and global_pruning_results:
        sel_dir = output_dir / "selectivity"
        print()
        print(f"Plotting per-model global pruning visualizations → {sel_dir}")
        for label, gp in global_pruning_results.items():
            safe_label     = label.replace(" ", "_").replace("=", "")
            vis_dir        = sel_dir / safe_label
            pruning_result = pruning_results.get(label)
            raw_t          = (pruning_result or {}).get("t_stats_per_layer") or gp.get("t_stats_per_layer")
            if not raw_t:
                print(f"  [skip] no t_stats_per_layer for {label}")
                continue
            t_stats_np = {int(k): np.array(v) for k, v in raw_t.items()}
            print(f"  {label}  →  {vis_dir.relative_to(output_dir)}/")
            save_global_selectivity_visualizations(
                t_stats_per_layer=t_stats_np,
                pruning_result=gp,
                label=label,
                vis_dir=vis_dir,
            )

    # ── Per-model rep-eng visualizations ────────────────────────────────────
    if not args.no_repeng and not args.no_selectivity and repeng_results:
        sel_dir = output_dir / "selectivity"
        print()
        print(f"Plotting per-model rep-eng visualizations → {sel_dir}")
        for label, tr in repeng_results.items():
            safe_label = label.replace(" ", "_").replace("=", "")
            vis_dir    = sel_dir / safe_label
            print(f"  {label}  →  {vis_dir.relative_to(output_dir)}/")
            save_repeng_visualizations(
                repeng_result=tr,
                label=label,
                vis_dir=vis_dir,
            )

    print()
    print("Done — all plots saved.")


if __name__ == "__main__":
    main()
