#!/usr/bin/env python3
"""
replot_toxicity_techniques_nanogpt.py
──────────────────────────────────────
Re-reads the JSON output files from eval_toxicity_techniques_nanogpt.py and
regenerates all plots.

Accepts individual paths or a directory containing `techniques_tau*.json` files.

Usage
-----
  # Re-plot everything in the default output directory
  python src/toxicity/replot_toxicity_techniques_nanogpt.py

  # Re-plot from a specific directory
  python src/toxicity/replot_toxicity_techniques_nanogpt.py \\
      --input_dir outputs/toxicity_techniques_nanogpt

  # Re-plot specific JSON files
  python src/toxicity/replot_toxicity_techniques_nanogpt.py \\
      --files outputs/toxicity_techniques_nanogpt/techniques_tau0_0.json \\
               outputs/toxicity_techniques_nanogpt/techniques_tau1_0.json

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
_EVAL = _HERE / "eval_toxicity_techniques_nanogpt.py"

if not _EVAL.exists():
    print(f"ERROR: cannot find eval script at {_EVAL}")
    sys.exit(1)

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("_eval_tech", str(_EVAL))
_mod  = _ilu.module_from_spec(_spec)
# We must NOT execute the module's `main()` — only expose its symbols.
# Override __name__ so the `if __name__ == "__main__": main()` guard fires.
_mod.__name__ = "_eval_tech_import"
try:
    _spec.loader.exec_module(_mod)   # type: ignore[union-attr]
except SystemExit:
    pass

# Pull in shared constants and plot functions
OUTPUT_DIR_DEFAULT  = _mod.OUTPUT_DIR
COLORS              = _mod.COLORS
METHOD_LABELS       = _mod.METHOD_LABELS
_METHOD_KEYS        = _mod._METHOD_KEYS
_DATASET_KEYS       = _mod._DATASET_KEYS
_DATASET_LABELS     = _mod._DATASET_LABELS
_tox_det            = _mod._tox_det
_tox_persp          = _mod._tox_persp
_ppl                = _mod._ppl
_vl                 = _mod._vl
_get_method_curve   = _mod._get_method_curve

plot_per_model_comparison  = _mod.plot_per_model_comparison
plot_cross_model_comparison = _mod.plot_cross_model_comparison
plot_dataset_comparison     = _mod.plot_dataset_comparison


# ── Additional replot-only visualisations ─────────────────────────────────────

def plot_bar_chart(
    sweep:       dict,
    dataset_key: str,
    label:       str,
    fracs:       list[float],
    output_dir:  Path,
    has_perspective: bool,
) -> None:
    """
    Bar chart comparing all 6 methods at each frac (plus baseline).
    One figure per dataset per model.  Groups: methods; x-axis ticks: fracs.
    """
    n_methods  = len(_METHOD_KEYS)
    bar_width  = 0.12
    x = np.arange(len(fracs) + 1)          # [0%  20%  50%] → indices 0 1 2
    x_labels   = ["0%"] + [f"{int(f*100)}%" for f in fracs]

    n_panels = 4 if has_perspective else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    ds_label  = _DATASET_LABELS.get(dataset_key, dataset_key)

    base = sweep.get("baseline", {})
    base_det   = base.get("detoxify",  {}).get("toxicity", {}).get("mean", float("nan"))
    base_persp = (base.get("perspective") or {}).get("toxicity", {}).get("mean", float("nan"))
    base_ppl   = base.get("perplexity", float("nan"))
    base_vl    = base.get("val_loss",   float("nan"))

    for mi, (mk, ml) in enumerate(zip(_METHOD_KEYS, METHOD_LABELS)):
        det_vals   = [base_det]   + _get_method_curve(sweep, mk, fracs, _tox_det)
        persp_vals = [base_persp] + _get_method_curve(sweep, mk, fracs, _tox_persp)
        ppl_vals   = [base_ppl]   + _get_method_curve(sweep, mk, fracs, _ppl)
        vl_vals    = [base_vl]    + _get_method_curve(sweep, mk, fracs, _vl)

        offset = (mi - n_methods / 2 + 0.5) * bar_width
        kw     = dict(color=COLORS[mi % len(COLORS)], width=bar_width,
                      label=ml, alpha=0.85)

        axes[0].bar(x + offset, det_vals, **kw)
        ax_off = 1
        if has_perspective:
            axes[1].bar(x + offset, persp_vals, **kw)
            ax_off = 2
        axes[ax_off].bar(x + offset, ppl_vals, **kw)
        axes[ax_off + 1].bar(x + offset, vl_vals, **kw)

    metric_labels = ["Mean toxicity (Detoxify)", "Perplexity", "Val Loss"]
    titles        = ["Detoxify Toxicity", "Perplexity", "Val Loss"]
    if has_perspective:
        metric_labels = ["Mean toxicity (Detoxify)", "Mean toxicity (Perspective)",
                         "Perplexity", "Val Loss"]
        titles        = ["Detoxify Toxicity", "Perspective Toxicity", "Perplexity", "Val Loss"]

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
    p = output_dir / f"bar_comparison_{safe}_{dataset_key}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


def plot_line_at_fracs(
    all_sweeps:  dict[str, dict],    # label → {dataset_key → sweep}
    target_frac: float,
    fracs:       list[float],
    output_dir:  Path,
    has_perspective: bool,
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
        n_panels  = 4 if has_perspective else 3
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

            det_vals   = [sweep.get(mk, {}).get(frac_str, {}).get("detoxify",   {})
                              .get("toxicity", {}).get("mean", float("nan"))
                          for mk in _METHOD_KEYS]
            persp_vals = [((sweep.get(mk, {}).get(frac_str, {}).get("perspective") or {})
                           .get("toxicity", {}).get("mean", float("nan")))
                          for mk in _METHOD_KEYS]
            ppl_vals   = [sweep.get(mk, {}).get(frac_str, {}).get("perplexity", float("nan"))
                          for mk in _METHOD_KEYS]
            vl_vals    = [sweep.get(mk, {}).get(frac_str, {}).get("val_loss",   float("nan"))
                          for mk in _METHOD_KEYS]

            offset = (li - n_models / 2 + 0.5) * bar_w
            col    = COLORS[li % len(COLORS)]
            kw     = dict(color=col, width=bar_w, label=lbl, alpha=0.85)

            axes[0].bar(x + offset, det_vals, **kw)
            ax_off = 1
            if has_perspective:
                axes[1].bar(x + offset, persp_vals, **kw)
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

        if has_perspective:
            ylabels = ["Tox (Detoxify)", "Tox (Perspective)", "Perplexity", "Val Loss"]
            titles  = ["Detoxify Tox", "Perspective Tox", "PPL", "Val Loss"]
        else:
            ylabels = ["Tox (Detoxify)", "Perplexity", "Val Loss"]
            titles  = ["Detoxify Tox", "PPL", "Val Loss"]

        for ax, yl, title in zip(axes, ylabels, titles):
            ax.set_ylabel(yl)
            ax.set_title(f"{title}  ·  {ds_label}  @  {frac_pct}", fontsize=10)

        fig.suptitle(f"All Methods @ {frac_pct}  ·  {ds_label}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        p = output_dir / f"methods_at_{frac_pct}_{ds_key}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {p}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _has_perspective(sweeps: dict) -> bool:
    for tau_data in sweeps.values():
        for ds_key in _DATASET_KEYS:
            if ds_key in tau_data:
                base = tau_data[ds_key].get("baseline", {})
                if base.get("perspective"):
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
    p.add_argument("--no_bar",       action="store_true", help="Skip bar charts")
    p.add_argument("--no_line",      action="store_true", help="Skip line-at-frac charts")
    p.add_argument("--no_per_model", action="store_true", help="Skip per-model line plots")
    p.add_argument("--no_cross",     action="store_true", help="Skip cross-model comparison")
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

    has_persp = _has_perspective(all_sweeps)
    print(f"Perspective data present: {has_persp}")
    print(f"Output directory: {out_dir}\n")

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
                    fracs=fracs, output_dir=out_dir, has_perspective=has_persp,
                )

            if not args.no_bar:
                plot_bar_chart(
                    sweep=sweep, dataset_key=ds_key, label=label,
                    fracs=fracs, output_dir=out_dir, has_perspective=has_persp,
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

    # ── Cross-model plots ─────────────────────────────────────────────────────
    if len(all_sweeps) > 1:
        if not args.no_cross:
            print("Cross-model comparison plots…")
            plot_cross_model_comparison(
                all_sweeps=all_sweeps, fracs=fracs,
                output_dir=out_dir, has_perspective=has_persp,
            )
            print()

    # ── Per-frac comparison bars ───────────────────────────────────────────────
    if not args.no_line:
        print("Generating per-frac comparison charts…")
        for frac in fracs:
            plot_line_at_fracs(
                all_sweeps=all_sweeps, target_frac=frac,
                fracs=fracs, output_dir=out_dir, has_perspective=has_persp,
            )
        print()

    print("All plots saved.")


if __name__ == "__main__":
    main()
