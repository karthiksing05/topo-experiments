#!/usr/bin/env python3
"""
replot_toxicity_techniques_nanogpt.py
──────────────────────────────────────
Re-reads the JSON output files from eval_toxicity_techniques_nanogpt_450m.py and
regenerates all plots for the 450M model.

Accepts individual paths or a directory containing `techniques_tau*.json` files.

Usage
-----
  # Re-plot everything in the default output directory
  python src/toxicity/replot_toxicity_techniques_nanogpt_450m.py

  # Re-plot from a specific directory
  python src/toxicity/replot_toxicity_techniques_nanogpt_450m.py \\
      --input_dir outputs/toxicity_techniques_nanogpt_450m

  # Save to a different directory
  python ... --output_dir custom_plots/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Import plotting helpers from eval script ─────────────────────────────────
# Resolve path relative to this file
_HERE = Path(__file__).resolve().parent
_EVAL = _HERE / "eval_toxicity_techniques_nanogpt_450m.py"

if not _EVAL.exists():
    print(f"ERROR: cannot find eval script at {_EVAL}")
    sys.exit(1)

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("_eval_tech", str(_EVAL))
_mod  = _ilu.module_from_spec(_spec)
# __name__ will be "_eval_tech" (not "__main__"), so the eval script's
# `if __name__ == "__main__": main()` guard will not fire.
_spec.loader.exec_module(_mod)   # type: ignore[union-attr]

# Pull in shared constants and plot functions
OUTPUT_DIR_DEFAULT  = _mod.OUTPUT_DIR
COLORS              = _mod.COLORS
METHOD_LABELS       = _mod.METHOD_LABELS
_METHOD_KEYS        = _mod._METHOD_KEYS
_DATASET_KEYS       = _mod._DATASET_KEYS
_DATASET_LABELS     = _mod._DATASET_LABELS
_tox_det            = _mod._tox_det
_tox_llamaguard     = _mod._tox_llamaguard
_ppl                = _mod._ppl
_vl                 = _mod._vl
_get_method_curve   = _mod._get_method_curve

_subdir                         = _mod._subdir

plot_per_model_comparison       = _mod.plot_per_model_comparison
plot_cross_model_comparison     = _mod.plot_cross_model_comparison
plot_dataset_comparison         = _mod.plot_dataset_comparison
plot_dual_toxicity_scorer       = _mod.plot_dual_toxicity_scorer
plot_selectivity_heatmap        = _mod.plot_selectivity_heatmap
plot_cortical_sheet_pruning     = _mod.plot_cortical_sheet_pruning
plot_global_pruning_bar         = _mod.plot_global_pruning_bar
plot_daa_osd_debug              = _mod.plot_daa_osd_debug
plot_daa_osd_cross_model        = _mod.plot_daa_osd_cross_model
plot_per_technique_cross_model  = _mod.plot_per_technique_cross_model


# ── Additional replot-only visualisations ─────────────────────────────────────

def plot_bar_chart(
    sweep:       dict,
    dataset_key: str,
    label:       str,
    fracs:       list[float],
    output_dir:  Path,
    has_llamaguard: bool,
) -> None:
    """
    Bar chart comparing all 6 methods at each frac (plus baseline).
    One figure per dataset per model.  Groups: methods; x-axis ticks: fracs.
    """
    n_methods  = len(_METHOD_KEYS)
    bar_width  = 0.12
    x = np.arange(len(fracs) + 1)
    x_labels   = ["0%"] + [f"{int(f*100)}%" for f in fracs]

    n_panels = 4 if has_llamaguard else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    ds_label  = _DATASET_LABELS.get(dataset_key, dataset_key)

    base = sweep.get("baseline", {})
    base_det = base.get("detoxify",   {}).get("toxicity", {}).get("mean", float("nan"))
    base_lg  = (base.get("llamaguard") or {}).get("toxicity", {}).get("mean", float("nan"))
    base_ppl = base.get("perplexity", float("nan"))
    base_vl  = base.get("val_loss",   float("nan"))

    for mi, (mk, ml) in enumerate(zip(_METHOD_KEYS, METHOD_LABELS)):
        det_vals = [base_det] + _get_method_curve(sweep, mk, fracs, _tox_det)
        lg_vals  = [base_lg]  + _get_method_curve(sweep, mk, fracs, _tox_llamaguard)
        ppl_vals = [base_ppl] + _get_method_curve(sweep, mk, fracs, _ppl)
        vl_vals  = [base_vl]  + _get_method_curve(sweep, mk, fracs, _vl)

        offset = (mi - n_methods / 2 + 0.5) * bar_width
        kw     = dict(color=COLORS[mi % len(COLORS)], width=bar_width,
                      label=ml, alpha=0.85)

        axes[0].bar(x + offset, det_vals, **kw)
        ax_off = 1
        if has_llamaguard:
            axes[1].bar(x + offset, lg_vals, **kw)
            ax_off = 2
        axes[ax_off].bar(x + offset, ppl_vals, **kw)
        axes[ax_off + 1].bar(x + offset, vl_vals, **kw)

    metric_labels = ["Mean toxicity (Detoxify)", "Perplexity", "Val Loss"]
    titles        = ["Detoxify Toxicity", "Perplexity", "Val Loss"]
    if has_llamaguard:
        metric_labels = ["Mean toxicity (Detoxify)", "Mean toxicity (Llama Guard)",
                         "Perplexity", "Val Loss"]
        titles        = ["Detoxify Toxicity", "Llama Guard Toxicity", "Perplexity", "Val Loss"]

    for ax, ylabel, title in zip(axes, metric_labels, titles):
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Intervention fraction")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}  ·  {ds_label}", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    safe = label.replace(" ", "_").replace("=", "").replace("(","").replace(")","")
    fig.suptitle(f"Method Comparison ({label})  ·  {ds_label}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = _subdir(output_dir, "bar_charts") / f"bar_comparison_{safe}_{dataset_key}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def plot_line_at_fracs(
    all_sweeps:  dict[str, dict],
    target_frac: float,
    fracs:       list[float],
    output_dir:  Path,
    has_llamaguard: bool,
) -> None:
    """
    Compare all models side-by-side at a SINGLE fraction point.
    One plot per dataset.  Methods on x-axis, bars per model.
    """
    if target_frac not in fracs:
        return
    frac_str  = str(target_frac)
    frac_pct  = f"{int(target_frac * 100)}%"

    for ds_key in _DATASET_KEYS:
        n_panels  = 4 if has_llamaguard else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        ds_label  = _DATASET_LABELS.get(ds_key, ds_key)

        labels    = list(all_sweeps.keys())
        n_models  = len(labels)
        bar_w     = 0.8 / max(n_models, 1)
        x         = np.arange(len(_METHOD_KEYS))

        for li, (lbl, model_data) in enumerate(all_sweeps.items()):
            sweep = model_data.get(ds_key, {})
            if not sweep:
                continue

            det_vals = [sweep.get(mk, {}).get(frac_str, {}).get("detoxify",   {})
                            .get("toxicity", {}).get("mean", float("nan"))
                        for mk in _METHOD_KEYS]
            lg_vals  = [((sweep.get(mk, {}).get(frac_str, {}).get("llamaguard") or {})
                         .get("toxicity", {}).get("mean", float("nan")))
                        for mk in _METHOD_KEYS]
            ppl_vals = [sweep.get(mk, {}).get(frac_str, {}).get("perplexity", float("nan"))
                        for mk in _METHOD_KEYS]
            vl_vals  = [sweep.get(mk, {}).get(frac_str, {}).get("val_loss",   float("nan"))
                        for mk in _METHOD_KEYS]

            offset = (li - n_models / 2 + 0.5) * bar_w
            col    = COLORS[li % len(COLORS)]
            kw     = dict(color=col, width=bar_w, label=lbl, alpha=0.85)

            axes[0].bar(x + offset, det_vals, **kw)
            ax_off = 1
            if has_llamaguard:
                axes[1].bar(x + offset, lg_vals, **kw)
                ax_off = 2
            axes[ax_off].bar(x + offset, ppl_vals, **kw)
            axes[ax_off + 1].bar(x + offset, vl_vals, **kw)

        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace(" ", "\n") for m in METHOD_LABELS],
                               fontsize=8)
            ax.set_ylim(bottom=0)
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(fontsize=7)

        if has_llamaguard:
            ylabels = ["Tox (Detoxify)", "Tox (Llama Guard)", "Perplexity", "Val Loss"]
            titles  = ["Detoxify Tox", "Llama Guard Tox", "PPL", "Val Loss"]
        else:
            ylabels = ["Tox (Detoxify)", "Perplexity", "Val Loss"]
            titles  = ["Detoxify Tox", "PPL", "Val Loss"]

        for ax, yl, title in zip(axes, ylabels, titles):
            ax.set_ylabel(yl)
            ax.set_title(f"{title}  ·  {ds_label}  @  {frac_pct}", fontsize=10)

        fig.suptitle(f"All Methods @ {frac_pct}  ·  {ds_label}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        p = _subdir(output_dir, "snapshot_fracs") / f"methods_at_{frac_pct}_{ds_key}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {p}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _has_llamaguard(sweeps: dict) -> bool:
    for tau_data in sweeps.values():
        for ds_key in _DATASET_KEYS:
            if ds_key in tau_data:
                base = tau_data[ds_key].get("baseline", {})
                if base.get("llamaguard"):
                    return True
    return False


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate plots from eval_toxicity_techniques_nanogpt.py output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_dir",  type=str, default=None,
                   help="Directory containing techniques_tau*.json files")
    p.add_argument("--files", nargs="*", default=None,
                   help="Explicit JSON file paths to read")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to save plots (defaults to input dir)")
    p.add_argument("--fracs",      type=str, default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5",
                   help="Comma-separated fracs (must match what was evaluated)")
    p.add_argument("--no_bar",            action="store_true", help="Skip bar charts")
    p.add_argument("--no_line",           action="store_true", help="Skip line-at-frac charts")
    p.add_argument("--no_per_model",      action="store_true", help="Skip per-model line plots")
    p.add_argument("--no_dual_scorer",    action="store_true",
                   help="Skip dual-scorer (Detoxify / Llama Guard) line plots")
    p.add_argument("--no_selectivity",    action="store_true",
                   help="Skip selectivity heatmaps (t-stat + pruning masks)")
    p.add_argument("--no_cortical",        action="store_true",
                   help="Skip cortical sheet + global pruning bar plots")
    p.add_argument("--no_daa_osd_debug",  action="store_true",
                   help="Skip per-model DAA/OSD debug figures")
    p.add_argument("--no_cross",          action="store_true", help="Skip cross-model comparison")
    p.add_argument("--no_per_technique",  action="store_true",
                   help="Skip per-technique cross-model plots")
    p.add_argument("--no_daa_osd_cross",  action="store_true",
                   help="Skip DAA/OSD cross-model comparison")
    p.add_argument("--no_dataset_compare", action="store_true",
                   help="Skip RTP vs ToxiGen comparison")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    fracs = [float(f) for f in args.fracs.split(",")]

    # ── Discover JSON files ────────────────────────────────────────────────────
    json_files: list[Path] = []
    if args.files:
        json_files = [Path(f) for f in args.files]
    else:
        search_dir = Path(args.input_dir) if args.input_dir else OUTPUT_DIR_DEFAULT
        json_files = sorted(search_dir.glob("techniques_tau*.json"))
        if not json_files:
            print(f"No techniques_tau*.json files found in {search_dir}")
            print("Pass --input_dir or --files to specify the source.")
            sys.exit(1)

    print(f"Found {len(json_files)} JSON file(s):")
    for jf in json_files:
        print(f"  {jf}")
    print()

    # ── Set output directory ──────────────────────────────────────────────────
    out_dir = Path(args.output_dir) if args.output_dir else json_files[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all files ────────────────────────────────────────────────────────
    all_sweeps: dict[str, dict] = {}   # label → tau_data
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        label = data.get("label") or f"tau={data.get('tau', '?')}"
        all_sweeps[label] = data

    has_llamaguard = _has_llamaguard(all_sweeps)
    print(f"LlamaGuard data present: {has_llamaguard}")
    print(f"Output directory: {out_dir}")

    # ── Discover which methods are available per model ────────────────────────
    # For cross-model plots we only include methods present in ALL models.
    _per_model_methods: dict[str, set[str]] = {}
    for label, tau_data in all_sweeps.items():
        methods_found: set[str] = set()
        for ds_key in _DATASET_KEYS:
            if ds_key not in tau_data:
                continue
            sweep = tau_data[ds_key]
            for mk in _METHOD_KEYS:
                if mk in sweep and sweep[mk]:
                    methods_found.add(mk)
        _per_model_methods[label] = methods_found

    # Methods present in ALL loaded models
    if _per_model_methods:
        _common_methods = set.intersection(*_per_model_methods.values())
    else:
        _common_methods = set()

    # Build filtered lists preserving original order
    _common_keys   = [mk for mk in _METHOD_KEYS if mk in _common_methods]
    _common_labels = [ml for mk, ml in zip(_METHOD_KEYS, METHOD_LABELS) if mk in _common_methods]
    _common_colors = [COLORS[i % len(COLORS)] for i, mk in enumerate(_METHOD_KEYS) if mk in _common_methods]

    print(f"Methods in all models ({len(_common_keys)}): {_common_keys}")
    for label, meths in _per_model_methods.items():
        only_here = meths - _common_methods
        if only_here:
            print(f"  {label} also has: {sorted(only_here)}")
    print()

    # ── Per-model plots ───────────────────────────────────────────────────────
    for label, tau_data in all_sweeps.items():
        print(f"── {label} ──")
        for ds_key in _DATASET_KEYS:
            if ds_key not in tau_data:
                continue
            sweep = tau_data[ds_key]

            if not args.no_per_model:
                plot_per_model_comparison(
                    sweep=sweep, dataset_key=ds_key, label=label,
                    fracs=fracs, output_dir=out_dir, has_llamaguard=has_llamaguard,
                )

            if not args.no_dual_scorer:
                plot_dual_toxicity_scorer(
                    sweep=sweep, dataset_key=ds_key, label=label,
                    fracs=fracs, output_dir=out_dir,
                )

            if not args.no_bar:
                plot_bar_chart(
                    sweep=sweep, dataset_key=ds_key, label=label,
                    fracs=fracs, output_dir=out_dir, has_llamaguard=has_llamaguard,
                )

            heur = sweep.get("heuristics", {})
            if not args.no_selectivity and heur:
                plot_selectivity_heatmap(
                    heuristics=heur, label=label,
                    dataset_key=ds_key, output_dir=out_dir,
                )

            if not args.no_cortical and heur:
                plot_cortical_sheet_pruning(
                    heuristics=heur, label=label,
                    dataset_key=ds_key, output_dir=out_dir,
                )
                plot_global_pruning_bar(
                    heuristics=heur, label=label,
                    dataset_key=ds_key, output_dir=out_dir,
                )

            if not args.no_daa_osd_debug and heur:
                plot_daa_osd_debug(
                    heuristics=heur, label=label,
                    dataset_key=ds_key, output_dir=out_dir,
                )

        if (not args.no_dataset_compare
                and "realtoxicityprompts" in tau_data
                and "toxigen" in tau_data):
            plot_dataset_comparison(
                model_label=label,
                rtp_sweep=tau_data["realtoxicityprompts"],
                tg_sweep=tau_data["toxigen"],
                fracs=fracs, output_dir=out_dir,
            )
        print()

    # ── Cross-model plots (only for methods present in ALL models) ───────────
    if len(all_sweeps) > 1 and _common_keys:
        if not args.no_cross:
            print("Cross-model comparison plots…")
            plot_cross_model_comparison(
                all_sweeps=all_sweeps, fracs=fracs,
                output_dir=out_dir, has_llamaguard=has_llamaguard,
                method_keys=_common_keys, method_labels=_common_labels,
            )
            print()
        if not args.no_per_technique:
            print("Per-technique cross-model plots…")
            for mk, ml in zip(_common_keys, _common_labels):
                plot_per_technique_cross_model(
                    all_sweeps=all_sweeps, method_key=mk, method_label=ml,
                    fracs=fracs, output_dir=out_dir, has_llamaguard=has_llamaguard,
                )
            print()

        if not args.no_daa_osd_cross:
            print("DAA/OSD cross-model debug plots…")
            for ds_key in _DATASET_KEYS:
                plot_daa_osd_cross_model(all_sweeps, ds_key, out_dir)
            print()
    # ── Per-frac comparison bars ───────────────────────────────────────────────
    if not args.no_line:
        print("Generating per-frac comparison charts…")
        for frac in fracs:
            plot_line_at_fracs(
                all_sweeps=all_sweeps, target_frac=frac,
                fracs=fracs, output_dir=out_dir, has_llamaguard=has_llamaguard,
            )
        print()

    print("All plots saved.")


if __name__ == "__main__":
    main()
