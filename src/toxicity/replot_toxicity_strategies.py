#!/usr/bin/env python3
"""
replot_toxicity_strategies.py
==============================
Unified replot that loads BOTH existing techniques JSON results AND new
strategies JSON results, then generates comparison plots across all methods.

Plot types:
  1. All-methods bar chart at a fixed frac
  2. Method-family grouped comparison
  3. Pareto frontier: toxicity vs perplexity
  4. Per-tau line plots (toxicity vs frac) for all methods overlaid
"""
from __future__ import annotations

import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Labels & families ─────────────────────────────────────────────────────────

# Existing techniques (from eval_toxicity_techniques_nanogpt.py)
TECHNIQUE_LABELS: dict[str, str] = {
    "per_layer_pruning":      "Per-layer Pruning",
    "global_pruning":         "Global Pruning",
    "per_layer_daa":          "Per-layer DAA",
    "global_daa":             "Global DAA",
    "per_layer_osd":          "Per-layer OSD",
    "global_osd":             "Global OSD",
    "topo_region_pruning":    "Topo Region Prune",
    "topo_smoothed_daa":      "Topo Smoothed DAA",
    "topo_spectral_cluster":  "Topo Spectral Cluster",
    "activation_steering":    "Activation Steering",
    "topo_lowrank_svd":       "Topo Lowrank SVD",
    "topo_freq_detox":        "Topo Freq Detox",
    "lowrank_toxic_projection": "Lowrank Toxic Proj",
}

TECHNIQUE_FAMILIES: dict[str, str] = {
    "per_layer_pruning":      "pruning",
    "global_pruning":         "pruning",
    "per_layer_daa":          "mean-diff",
    "global_daa":             "mean-diff",
    "per_layer_osd":          "subspace",
    "global_osd":             "subspace",
    "topo_region_pruning":    "topo",
    "topo_smoothed_daa":      "topo",
    "topo_spectral_cluster":  "topo",
    "activation_steering":    "steering",
    "topo_lowrank_svd":       "topo",
    "topo_freq_detox":        "topo",
    "lowrank_toxic_projection": "subspace",
}

# New strategies (from eval_toxicity_strategies_nanogpt.py)
STRATEGY_LABELS: dict[str, str] = {
    "eigenshift":        "EigenShift",
    "self_debiasing":    "Self-Debiasing",
    "chars_lite":        "CHaRS-lite",
    "vocab_shifting":    "Vocab Shifting",
    "pct_osd":           "PCT (thresh. OSD)",
}

STRATEGY_FAMILIES: dict[str, str] = {
    "eigenshift":        "mechanistic",
    "self_debiasing":    "decoding",
    "chars_lite":        "mechanistic",
    "vocab_shifting":    "decoding",
    "pct_osd":           "mechanistic",
}

ALL_LABELS   = {**TECHNIQUE_LABELS, **STRATEGY_LABELS}
ALL_FAMILIES = {**TECHNIQUE_FAMILIES, **STRATEGY_FAMILIES}

FAMILY_DISPLAY = {
    "pruning":     "Pruning",
    "mean-diff":   "Mean-Difference",
    "subspace":    "Subspace Projection",
    "topo":        "Topology-Aware",
    "steering":    "Activation Steering",
    "mechanistic": "Mechanistic (new)",
    "decoding":    "Decoding-Time (new)",
}

FAMILY_COLORS = {
    "pruning":     "#1f77b4",
    "mean-diff":   "#ff7f0e",
    "subspace":    "#2ca02c",
    "topo":        "#d62728",
    "steering":    "#9467bd",
    "mechanistic": "#8c564b",
    "decoding":    "#e377c2",
}

_DS_KEYS = ["realtoxicityprompts", "toxigen"]
_DS_LABELS = {
    "realtoxicityprompts": "RealToxicityPrompts",
    "toxigen": "ToxiGen",
}


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def _load_jsons(globpattern: str, directory: Path) -> dict[str, dict]:
    """Load matching JSONs from directory, keyed by their label field."""
    out = {}
    for jf in sorted(directory.glob(globpattern)):
        with open(jf) as f:
            data = json.load(f)
        label = data.get("label") or f"tau={data.get('tau', '?')}"
        out[label] = data
    return out


def _extract_tox(entry: dict) -> float:
    try:
        return entry["detoxify"]["toxicity"]["mean"]
    except (KeyError, TypeError):
        return float("nan")


def _extract_ppl(entry: dict) -> float:
    try:
        return entry["perplexity"]
    except (KeyError, TypeError):
        return float("nan")


def _extract_ppl_ratio(entry: dict) -> float:
    try:
        return entry["ppl_ratio"]
    except (KeyError, TypeError):
        return float("nan")


def _subdir(base: Path, name: str) -> Path:
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ═════════════════════════════════════════════════════════════════════════════
# MERGE  — combine techniques + strategies data per tau
# ═════════════════════════════════════════════════════════════════════════════

def merge_sweeps(
    techniques: dict[str, dict],
    strategies: dict[str, dict],
) -> dict[str, dict]:
    """Merge techniques and strategies JSON dicts.

    For each matching tau label, combine per-dataset method dicts.
    Returns the merged dict keyed by label."""
    merged: dict[str, dict] = {}

    for label, tech_data in techniques.items():
        merged[label] = dict(tech_data)

    for label, strat_data in strategies.items():
        if label not in merged:
            merged[label] = dict(strat_data)
        else:
            for ds_key in _DS_KEYS:
                if ds_key not in strat_data:
                    continue
                if ds_key not in merged[label]:
                    merged[label][ds_key] = strat_data[ds_key]
                else:
                    # Merge method-level keys (eigenshift, etc.)
                    for mk, mv in strat_data[ds_key].items():
                        if mk in ("fracs", "baseline", "heuristics"):
                            continue
                        merged[label][ds_key][mk] = mv
                    # Merge fracs if missing
                    if "fracs" not in merged[label][ds_key]:
                        merged[label][ds_key]["fracs"] = strat_data[ds_key].get("fracs")

    return merged


def _available_methods(sweep: dict, ds_key: str) -> list[str]:
    """Return method keys present in sweep[ds_key] with actual non-empty data."""
    ds = sweep.get(ds_key, {})
    methods = []
    for mk in list(TECHNIQUE_LABELS) + list(STRATEGY_LABELS):
        if mk in ds and ds[mk]:
            methods.append(mk)
    return methods


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1: ALL-METHODS BAR CHART
# ═════════════════════════════════════════════════════════════════════════════

def plot_all_methods_bar(
    sweep: dict,
    ds_key: str,
    label: str,
    target_frac: float,
    output_dir: Path,
) -> None:
    """Bar chart at a fixed frac: toxicity and perplexity ratio for each
    method, grouped side-by-side."""
    ds = sweep.get(ds_key, {})
    baseline = ds.get("baseline", {})
    base_tox = _extract_tox(baseline)

    methods = _available_methods(sweep, ds_key)
    if not methods:
        return

    frac_str = str(target_frac)
    names, tox_vals, ppl_ratios, colors = [], [], [], []

    for mk in methods:
        md = ds.get(mk, {})
        entry = md.get(frac_str)
        if not entry:
            continue
        names.append(ALL_LABELS.get(mk, mk))
        tox_vals.append(_extract_tox(entry))
        ppl_ratios.append(_extract_ppl_ratio(entry))
        fam = ALL_FAMILIES.get(mk, "mechanistic")
        colors.append(FAMILY_COLORS.get(fam, "#999999"))

    if not names:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, len(names)*0.7), 6))
    x = np.arange(len(names))
    w = 0.55

    # --- Toxicity ---
    bars1 = ax1.bar(x, tox_vals, w, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(base_tox, color="red", ls="--", lw=1.2, label=f"Baseline ({base_tox:.4f})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=55, ha="right", fontsize=8)
    ax1.set_ylabel("Mean Toxicity (Detoxify)")
    ax1.set_title(f"Toxicity @ frac={target_frac}")
    ax1.legend(fontsize=8)

    # --- PPL Ratio ---
    bars2 = ax2.bar(x, ppl_ratios, w, color=colors, edgecolor="black", linewidth=0.5)
    ax2.axhline(1.0, color="red", ls="--", lw=1.2, label="Baseline (1.0)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=55, ha="right", fontsize=8)
    ax2.set_ylabel("Perplexity Ratio (method/baseline)")
    ax2.set_title(f"Perplexity Ratio @ frac={target_frac}")
    ax2.legend(fontsize=8)

    fig.suptitle(f"{_DS_LABELS.get(ds_key, ds_key)} · {label}", fontsize=12, y=1.02)
    fig.tight_layout()

    safe = label.replace("=", "").replace(" ", "_")
    out = _subdir(output_dir, "bar_all_methods")
    fig.savefig(out / f"bar_all_{safe}_{ds_key}_frac{target_frac}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved bar chart → {out.name}/")


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2: FAMILY-GROUPED COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

def plot_family_grouped(
    sweep: dict,
    ds_key: str,
    label: str,
    target_frac: float,
    output_dir: Path,
) -> None:
    """Grouped bar chart: methods grouped by family (pruning, subspace, decoding, …)."""
    ds = sweep.get(ds_key, {})
    baseline = ds.get("baseline", {})
    base_tox = _extract_tox(baseline)

    methods = _available_methods(sweep, ds_key)
    frac_str = str(target_frac)

    # Group by family
    families: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for mk in methods:
        entry = ds.get(mk, {}).get(frac_str)
        if not entry:
            continue
        fam = ALL_FAMILIES.get(mk, "other")
        families[fam].append((
            ALL_LABELS.get(mk, mk),
            _extract_tox(entry),
            _extract_ppl_ratio(entry),
        ))

    if not families:
        return

    # Order families for display
    fam_order = ["pruning", "mean-diff", "subspace", "steering", "topo",
                 "mechanistic", "decoding"]
    fam_order = [f for f in fam_order if f in families]
    for f in families:
        if f not in fam_order:
            fam_order.append(f)

    fig, ax = plt.subplots(figsize=(max(14, sum(len(families[f]) for f in fam_order)*0.7 + len(fam_order)*0.5), 6))

    x_pos = 0
    xticks, xlabels = [], []
    group_spans = []  # (start, end, family_name)

    for fi, fam in enumerate(fam_order):
        members = families[fam]
        start = x_pos
        for name, tox, ppl_r in members:
            color = FAMILY_COLORS.get(fam, "#999999")
            ax.bar(x_pos, tox, 0.6, color=color, edgecolor="black", linewidth=0.5)
            xticks.append(x_pos)
            xlabels.append(name)
            x_pos += 1
        group_spans.append((start, x_pos - 1, fam))
        x_pos += 0.5  # gap between families

    ax.axhline(base_tox, color="red", ls="--", lw=1.2, label=f"Baseline ({base_tox:.4f})")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Mean Toxicity (Detoxify)")
    ax.set_title(f"Family-Grouped @ frac={target_frac} · {_DS_LABELS.get(ds_key, ds_key)} · {label}")

    # Add family-group brackets
    for start, end, fam in group_spans:
        mid = (start + end) / 2
        ax.annotate(FAMILY_DISPLAY.get(fam, fam),
                     xy=(mid, 0), xytext=(mid, -0.06),
                     textcoords="axes fraction",
                     ha="center", fontsize=7, style="italic",
                     arrowprops=dict(arrowstyle="-", lw=0))

    # Legend
    handles = [mpatches.Patch(color=FAMILY_COLORS.get(f, "#999"), label=FAMILY_DISPLAY.get(f, f))
               for f in fam_order]
    ax.legend(handles=handles, fontsize=7, ncol=min(4, len(handles)),
              loc="upper right")

    fig.tight_layout()
    safe = label.replace("=", "").replace(" ", "_")
    out = _subdir(output_dir, "family_grouped")
    fig.savefig(out / f"family_{safe}_{ds_key}_frac{target_frac}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved family grouped → {out.name}/")


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3: PARETO FRONTIER
# ═════════════════════════════════════════════════════════════════════════════

def plot_pareto_frontier(
    sweep: dict,
    ds_key: str,
    label: str,
    target_frac: float,
    output_dir: Path,
) -> None:
    """Scatter: toxicity (x) vs perplexity ratio (y).
    Connect Pareto-optimal methods."""
    ds = sweep.get(ds_key, {})
    baseline = ds.get("baseline", {})
    base_tox = _extract_tox(baseline)
    frac_str = str(target_frac)

    methods = _available_methods(sweep, ds_key)
    points = []  # (tox, ppl_ratio, name, family)
    for mk in methods:
        entry = ds.get(mk, {}).get(frac_str)
        if not entry:
            continue
        tox = _extract_tox(entry)
        ppl_r = _extract_ppl_ratio(entry)
        if math.isnan(tox) or math.isnan(ppl_r):
            continue
        points.append((tox, ppl_r, ALL_LABELS.get(mk, mk), ALL_FAMILIES.get(mk, "other")))

    if not points:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter
    for tox, ppl_r, name, fam in points:
        color = FAMILY_COLORS.get(fam, "#999999")
        ax.scatter(tox, ppl_r, c=color, s=80, edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(name, (tox, ppl_r), fontsize=6, ha="left", va="bottom",
                     xytext=(4, 4), textcoords="offset points")

    # Baseline point
    ax.scatter(base_tox, 1.0, c="red", s=120, marker="*", zorder=4, label="Baseline")

    # Pareto front: want lower toxicity AND lower ppl_ratio
    pts_arr = np.array([(t, p) for t, p, _, _ in points])
    # Sort by tox ascending
    order = np.argsort(pts_arr[:, 0])
    pareto = []
    best_ppl = float("inf")
    for idx in order:
        if pts_arr[idx, 1] <= best_ppl:
            pareto.append(idx)
            best_ppl = pts_arr[idx, 1]

    if len(pareto) >= 2:
        pareto_pts = pts_arr[pareto]
        ax.plot(pareto_pts[:, 0], pareto_pts[:, 1], 'k--', alpha=0.5,
                lw=1.5, label="Pareto frontier")

    ax.set_xlabel("Mean Toxicity (Detoxify) ↓ better")
    ax.set_ylabel("Perplexity Ratio ↓ better")
    ax.set_title(f"Pareto Frontier @ frac={target_frac} · "
                 f"{_DS_LABELS.get(ds_key, ds_key)} · {label}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    safe = label.replace("=", "").replace(" ", "_")
    out = _subdir(output_dir, "pareto")
    fig.savefig(out / f"pareto_{safe}_{ds_key}_frac{target_frac}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved Pareto frontier → {out.name}/")


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 4: LINE PLOT (toxicity vs frac, all methods overlaid)
# ═════════════════════════════════════════════════════════════════════════════

def plot_tox_vs_frac(
    sweep: dict,
    ds_key: str,
    label: str,
    fracs: list[float],
    output_dir: Path,
) -> None:
    """Line plot: toxicity vs intervention fraction for every method."""
    ds = sweep.get(ds_key, {})
    baseline = ds.get("baseline", {})
    base_tox = _extract_tox(baseline)

    methods = _available_methods(sweep, ds_key)
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(base_tox, color="red", ls="--", lw=1, alpha=0.6, label="Baseline")

    for mi, mk in enumerate(methods):
        md = ds.get(mk, {})
        tox_vals = []
        for frac in fracs:
            entry = md.get(str(frac))
            tox_vals.append(_extract_tox(entry) if entry else float("nan"))
        fam = ALL_FAMILIES.get(mk, "other")
        color = FAMILY_COLORS.get(fam, "#999")
        is_strategy = mk in STRATEGY_LABELS
        ax.plot(fracs, tox_vals, marker="o" if is_strategy else "s",
                markersize=4, lw=1.5 if is_strategy else 1.0,
                ls="-" if is_strategy else "--",
                color=color, alpha=0.85,
                label=ALL_LABELS.get(mk, mk))

    ax.set_xlabel("Intervention Fraction")
    ax.set_ylabel("Mean Toxicity (Detoxify)")
    ax.set_title(f"Toxicity vs Fraction · {_DS_LABELS.get(ds_key, ds_key)} · {label}")
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    safe = label.replace("=", "").replace(" ", "_")
    out = _subdir(output_dir, "tox_vs_frac")
    fig.savefig(out / f"tox_vs_frac_{safe}_{ds_key}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved tox-vs-frac → {out.name}/")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Unified replot: merge techniques + strategies results and "
                    "produce comparison plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--techniques_dir", type=str, default=None,
                   help="Directory with techniques_tau*.json files")
    p.add_argument("--strategies_dir", type=str, default=None,
                   help="Directory with strategies_tau*.json files")
    p.add_argument("--variant", type=str, choices=["125m", "450m"], default="125m",
                   help="Model variant (sets default directories)")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--fracs", type=str,
                   default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
    p.add_argument("--target_fracs", type=str, default="0.1,0.2,0.3",
                   help="Fracs at which to produce bar/Pareto plots")
    p.add_argument("--no_bar", action="store_true")
    p.add_argument("--no_family", action="store_true")
    p.add_argument("--no_pareto", action="store_true")
    p.add_argument("--no_line", action="store_true")
    args = p.parse_args()

    # Resolve directories
    if args.variant == "450m":
        tech_dir = Path(args.techniques_dir or "outputs/toxicity_techniques_nanogpt_450m")
        strat_dir = Path(args.strategies_dir or "outputs/toxicity_strategies_nanogpt_450m")
        default_out = "outputs/toxicity_comparison_nanogpt_450m"
    else:
        tech_dir = Path(args.techniques_dir or "outputs/toxicity_techniques_nanogpt")
        strat_dir = Path(args.strategies_dir or "outputs/toxicity_strategies_nanogpt")
        default_out = "outputs/toxicity_comparison_nanogpt"

    output_dir = Path(args.output_dir or default_out)
    output_dir.mkdir(parents=True, exist_ok=True)

    fracs = [float(f) for f in args.fracs.split(",")]
    target_fracs = [float(f) for f in args.target_fracs.split(",")]

    # Load
    techniques = _load_jsons("techniques_tau*.json", tech_dir) if tech_dir.is_dir() else {}
    strategies = _load_jsons("strategies_tau*.json", strat_dir) if strat_dir.is_dir() else {}

    print(f"Loaded {len(techniques)} technique files, {len(strategies)} strategy files")
    if not techniques and not strategies:
        print("No data found. Check --techniques_dir and --strategies_dir.")
        return

    merged = merge_sweeps(techniques, strategies)
    print(f"Merged into {len(merged)} tau labels: {list(merged.keys())}")

    # Generate plots
    for label, tau_data in merged.items():
        print(f"\n── {label} ──")
        for ds_key in _DS_KEYS:
            if ds_key not in tau_data:
                continue
            print(f"  Dataset: {_DS_LABELS.get(ds_key, ds_key)}")
            methods = _available_methods(tau_data, ds_key)
            print(f"    Methods ({len(methods)}): {methods}")

            if not args.no_line:
                plot_tox_vs_frac(tau_data, ds_key, label, fracs, output_dir)

            for tf in target_fracs:
                if not args.no_bar:
                    plot_all_methods_bar(tau_data, ds_key, label, tf, output_dir)
                if not args.no_family:
                    plot_family_grouped(tau_data, ds_key, label, tf, output_dir)
                if not args.no_pareto:
                    plot_pareto_frontier(tau_data, ds_key, label, tf, output_dir)

    print(f"\nAll plots saved → {output_dir}/")


if __name__ == "__main__":
    main()
