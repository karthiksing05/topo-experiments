#!/usr/bin/env python3
"""compare_all_methods_toxicity.py

For every model (tau value) found in an output directory, produces a
3-panel figure comparing ALL available intervention methods:

    [ Toxicity (y ≤ 0.15) | Perplexity (y ≤ 10) | Val Loss (y ≤ 6) ]

Each method is a separate line.  The x-axis is normalised so that
  x = 0  →  baseline (unmodified model)
  x = 1  →  strongest sweep point in that method's sweep

Methods discovered automatically from *_tau*.json files:
  pruning_tau*            — per-layer neuron (t-stat) pruning
  global_pruning_tau*     — cross-layer global neuron pruning
  svd_pruning_tau*        — SVD-based selective pruning
  pca_pruning_tau*        — PCA active-subspace pruning
  daa_pruning_tau*        — Differential Activation Analysis pruning
  osd_pruning_tau*        — Orthogonal Subspace Decomposition pruning
  attenuation_tau*        — toxic-neuron attenuation
  amplification_tau*      — toxic-neuron amplification
  repeng_tau*             — representation-engineering steering

Usage
-----
  python compare_all_methods_toxicity.py [--output_dir PATH] [--out_subdir NAME]

Plots are written to <output_dir>/<out_subdir>/  (default: all_methods_comparison/).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR            = Path(__file__).resolve().parents[2]
OUTPUT_DIR_DEFAULT  = BASE_DIR / "outputs" / "toxicity_nanogpt"

# ── Constants ──────────────────────────────────────────────────────────────────
TOX_MAX     = 0.15   # y-axis ceiling for the toxicity panel
PPL_MAX     = 10.0   # y-axis ceiling for the perplexity panel
VAL_LOSS_MAX = 6.0   # y-axis ceiling for the val-loss panel

COLORS = [
    "#1f77b4",   # blue         — neuron pruning
    "#ff7f0e",   # orange       — global pruning
    "#9467bd",   # purple       — DAA pruning
    "#8c564b",   # brown        — OSD pruning
    "#e377c2",   # pink         — attenuation
]

# ── Method registry ────────────────────────────────────────────────────────────
# (display_name, json_stem_prefix, fracs_key, results_key)
#   fracs_key   — JSON key whose value is the list of sweep parameter values
#   results_key — JSON key whose value is the per-step results dict
METHODS: list[tuple[str, str, str, str]] = [
    ("Neuron pruning", "pruning",        "pruning_fractions", "pruned"),
    ("Global pruning", "global_pruning", "pruning_fractions", "pruned"),
    ("DAA pruning",    "daa_pruning",    "pruning_fractions", "pruned"),
    ("OSD pruning",    "osd_pruning",    "pruning_fractions", "pruned"),
    ("Attenuation",    "attenuation",    "att_fracs",         "attenuated"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_json(p: Path) -> dict | None:
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _tox_ppl_vl(d: dict) -> tuple[float, float, float]:
    """Return (mean_toxicity, perplexity, val_loss) from a result sub-dict."""
    tox = d.get("toxicity_scores", {}).get("toxicity", {}).get("mean", float("nan"))
    ppl = d.get("perplexity", d.get("ppl", float("nan")))
    vl  = d.get("val_loss", float("nan"))
    return tox, ppl, vl


def _extract_sweep(
    data:        dict,
    fracs_key:   str,
    results_key: str,
) -> tuple[list[float], list[float], list[float], list[float],
           float, float, float] | None:
    """
    Parse a sweep JSON and return
      (xs_norm, tox_vals, ppl_vals, vl_vals, base_tox, base_ppl, base_vl)

    xs_norm is evenly spaced in (0, 1] with n points where n = len(sweep).
    The caller should prepend x=0 and the baseline values to get the full line.
    """
    fracs   = data.get(fracs_key, [])
    results = data.get(results_key, {})
    base    = data.get("unpruned", {})

    if not fracs or not results:
        return None

    bt, bp, bv = _tox_ppl_vl(base)

    tox_vals, ppl_vals, vl_vals = [], [], []
    for f in fracs:
        # Try str(f) then str(float(f)) to handle minor formatting differences
        rd = results.get(str(f))
        if rd is None:
            for k in results:
                try:
                    if abs(float(k) - float(f)) < 1e-9:
                        rd = results[k]
                        break
                except ValueError:
                    pass
        if rd is None:
            tox_vals.append(float("nan"))
            ppl_vals.append(float("nan"))
            vl_vals.append(float("nan"))
        else:
            t, p, v = _tox_ppl_vl(rd)
            tox_vals.append(t)
            ppl_vals.append(p)
            vl_vals.append(v)

    n = len(fracs)
    xs_norm = [(i + 1) / n for i in range(n)]   # 1/n, 2/n, … 1.0

    return xs_norm, tox_vals, ppl_vals, vl_vals, bt, bp, bv


def _build_tau_label_map(output_dir: Path) -> dict[str, str]:
    """Map safe_tau strings (e.g. "0_0") → model labels (e.g. "tau=0.0 (baseline)")."""
    tau_to_label: dict[str, str] = {}
    results_path = output_dir / "results.json"
    if results_path.exists():
        data = _load_json(results_path) or {}
        for label in data:
            for part in label.split():
                if part.startswith("tau="):
                    tau_str = part.split("=")[1]
                    tau_to_label[tau_str] = label
                    break
    if not tau_to_label:
        # Fallback: infer from filenames
        for p in sorted(output_dir.glob("pruning_tau*.json")):
            stem    = p.stem.removeprefix("pruning_tau")
            tau_str = stem.replace("_", ".")
            tau_to_label[tau_str] = f"tau={tau_str}"
    return tau_to_label


def _discover_safe_taus(output_dir: Path) -> list[str]:
    """Collect every unique safe_tau found across any *_tau*.json file."""
    taus: set[str] = set()
    for p in output_dir.glob("*_tau*.json"):
        name = p.stem
        idx  = name.rfind("_tau")
        if idx == -1:
            continue
        safe_tau = name[idx + len("_tau"):]
        # Sanity check: should look like digits and underscores only
        if all(c.isdigit() or c in "_." for c in safe_tau):
            taus.add(safe_tau)
    return sorted(taus)


# ── Per-model plot ─────────────────────────────────────────────────────────────

def _format_safe(label: str) -> str:
    return label.replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")


def plot_model_comparison(
    label:      str,
    safe_tau:   str,
    output_dir: Path,
    out_dir:    Path,
) -> None:
    """
    Load every available method JSON for *safe_tau*, then save a 3-panel
    (toxicity | perplexity | val_loss) PNG to *out_dir*.
    """
    # ── Collect sweep data ────────────────────────────────────────────────────
    sweeps: list[tuple[str, str, tuple]] = []
    for (name, prefix, fracs_key, results_key), color in zip(METHODS, COLORS):
        json_path = output_dir / f"{prefix}_tau{safe_tau}.json"
        if not json_path.exists():
            continue
        data = _load_json(json_path)
        if data is None:
            continue
        sweep = _extract_sweep(data, fracs_key, results_key)
        if sweep is not None:
            sweeps.append((name, color, sweep))
            print(f"    [{name}]")

    if not sweeps:
        print(f"  No method data found for {label} — skipping")
        return

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, (ax_tox, ax_ppl, ax_vl) = plt.subplots(1, 3, figsize=(18, 5))

    for name, color, (xs, tox, ppl, vl, bt, bp, bv) in sweeps:
        x_full   = [0.0] + list(xs)
        tox_full = [bt]  + list(tox)
        ppl_full = [bp]  + list(ppl)
        vl_full  = [bv]  + list(vl)

        kw = dict(color=color, linewidth=2, markersize=5)
        ax_tox.plot(x_full, tox_full, "o-", label=name, **kw)
        ax_ppl.plot(x_full, ppl_full, "s-", label=name, **kw)
        ax_vl.plot(x_full,  vl_full,  "^-", label=name, **kw)

    # ── Toxicity panel ────────────────────────────────────────────────────────
    ax_tox.set_xlabel("Normalized intervention strength")
    ax_tox.set_ylabel("Mean toxicity")
    ax_tox.set_title(f"Toxicity  (truncated at {TOX_MAX})")
    ax_tox.set_xlim(-0.03, 1.05)
    ax_tox.set_ylim(0, TOX_MAX)
    ax_tox.grid(True, alpha=0.3)
    ax_tox.legend(fontsize=8)

    # ── Perplexity panel (linear, truncated) ──────────────────────────────────
    ax_ppl.set_xlabel("Normalized intervention strength")
    ax_ppl.set_ylabel("Perplexity")
    ax_ppl.set_title(f"Perplexity  (truncated at {PPL_MAX:.0f})")
    ax_ppl.set_xlim(-0.03, 1.05)
    ax_ppl.set_ylim(0, PPL_MAX)
    ax_ppl.grid(True, alpha=0.3)
    ax_ppl.legend(fontsize=8)

    # ── Val-loss panel (y truncated at VAL_LOSS_MAX) ──────────────────────────
    ax_vl.set_xlabel("Normalized intervention strength")
    ax_vl.set_ylabel("Validation loss")
    ax_vl.set_title(f"Val Loss  (truncated at {VAL_LOSS_MAX:.0f})")
    ax_vl.set_xlim(-0.03, 1.05)
    ax_vl.set_ylim(0, VAL_LOSS_MAX)
    ax_vl.grid(True, alpha=0.3)
    ax_vl.legend(fontsize=8)

    fig.suptitle(f"All Methods — {label}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = out_dir / f"all_methods_{_format_safe(label)}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── Combined summary grid (all models × 3 panels) ─────────────────────────────

def plot_combined_summary(
    tau_sweeps: list[tuple[str, list[tuple[str, str, tuple]]]],
    out_dir:    Path,
) -> None:
    """
    Rows = models, cols = [toxicity | perplexity | val_loss].
    One figure saved as all_methods_summary.png.
    """
    n_models = len(tau_sweeps)
    if n_models == 0:
        return

    fig, axes = plt.subplots(
        n_models, 3,
        figsize=(18, 4 * n_models),
        squeeze=False,
    )

    for row, (label, sweeps) in enumerate(tau_sweeps):
        ax_tox, ax_ppl, ax_vl = axes[row]

        for name, color, (xs, tox, ppl, vl, bt, bp, bv) in sweeps:
            x_full   = [0.0] + list(xs)
            tox_full = [bt]  + list(tox)
            ppl_full = [bp]  + list(ppl)
            vl_full  = [bv]  + list(vl)

            kw = dict(color=color, linewidth=1.8, markersize=4)
            ax_tox.plot(x_full, tox_full, "o-", label=name, **kw)
            ax_ppl.plot(x_full, ppl_full, "s-", label=name, **kw)
            ax_vl.plot(x_full,  vl_full,  "^-", label=name, **kw)

        for ax in (ax_tox, ax_ppl, ax_vl):
            ax.set_xlim(-0.03, 1.05)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Norm. strength", fontsize=8)

        ax_tox.set_ylabel("Mean toxicity", fontsize=8)
        ax_tox.set_title(f"{label}  |  Toxicity", fontsize=9)
        ax_tox.set_ylim(0, TOX_MAX)

        ax_ppl.set_ylabel("Perplexity", fontsize=8)
        ax_ppl.set_title(f"{label}  |  Perplexity (trunc @{PPL_MAX:.0f})", fontsize=9)
        ax_ppl.set_ylim(0, PPL_MAX)

        ax_vl.set_ylabel("Val loss", fontsize=8)
        ax_vl.set_title(f"{label}  |  Val Loss (trunc @{VAL_LOSS_MAX:.0f})", fontsize=9)
        ax_vl.set_ylim(0, VAL_LOSS_MAX)

        # Legend only on the first row to avoid clutter
        if row == 0:
            for ax in (ax_tox, ax_ppl, ax_vl):
                ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("All Methods × All Models", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = out_dir / "all_methods_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary grid → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-model comparison of all toxicity intervention methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output_dir", type=str, default=None,
        help=f"Directory containing *_tau*.json result files  (default: {OUTPUT_DIR_DEFAULT})",
    )
    p.add_argument(
        "--out_subdir", type=str, default="all_methods_comparison",
        help="Subdirectory under output_dir where plots are saved",
    )
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR_DEFAULT

    if not output_dir.is_dir():
        raise SystemExit(f"Output directory not found: {output_dir}")

    out_dir = output_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory : {output_dir}")
    print(f"Plots saved to   : {out_dir}")
    print()

    tau_to_label = _build_tau_label_map(output_dir)
    safe_taus    = _discover_safe_taus(output_dir)

    if not safe_taus:
        raise SystemExit(f"No *_tau*.json files found in {output_dir}")

    tau_labels = [
        tau_to_label.get(st.replace("_", "."), f"tau={st.replace('_', '.')}")
        for st in safe_taus
    ]
    print(f"Found {len(safe_taus)} model(s): {tau_labels}")
    print()

    # ── Per-model plots ───────────────────────────────────────────────────
    all_sweeps: list[tuple[str, list]] = []

    for safe_tau, label in zip(safe_taus, tau_labels):
        print(f"=== {label} ===")

        # Collect sweeps for this tau (needed for summary grid too)
        sweeps: list[tuple[str, str, tuple]] = []
        for (name, prefix, fracs_key, results_key), color in zip(METHODS, COLORS):
            json_path = output_dir / f"{prefix}_tau{safe_tau}.json"
            if not json_path.exists():
                continue
            data = _load_json(json_path)
            if data is None:
                continue
            sweep = _extract_sweep(data, fracs_key, results_key)
            if sweep is not None:
                sweeps.append((name, color, sweep))
                print(f"    [{name}]")

        if not sweeps:
            print(f"  No method data found — skipping")
        else:
            # Per-model 3-panel figure
            fig, (ax_tox, ax_ppl, ax_vl) = plt.subplots(1, 3, figsize=(18, 5))

            for name, color, (xs, tox, ppl, vl, bt, bp, bv) in sweeps:
                x_full   = [0.0] + list(xs)
                tox_full = [bt]  + list(tox)
                ppl_full = [bp]  + list(ppl)
                vl_full  = [bv]  + list(vl)

                kw = dict(color=color, linewidth=2, markersize=5)
                ax_tox.plot(x_full, tox_full, "o-", label=name, **kw)
                ax_ppl.plot(x_full, ppl_full, "s-", label=name, **kw)
                ax_vl.plot(x_full,  vl_full,  "^-", label=name, **kw)

            # Toxicity
            ax_tox.set_xlabel("Normalized intervention strength")
            ax_tox.set_ylabel("Mean toxicity")
            ax_tox.set_title("Toxicity")
            ax_tox.set_xlim(-0.03, 1.05)
            ax_tox.set_ylim(0, TOX_MAX)
            ax_tox.grid(True, alpha=0.3)
            ax_tox.legend(fontsize=8)

            # Perplexity (linear, truncated)
            ax_ppl.set_xlabel("Normalized intervention strength")
            ax_ppl.set_ylabel("Perplexity")
            ax_ppl.set_title(f"Perplexity  (truncated at {PPL_MAX:.0f})")
            ax_ppl.set_xlim(-0.03, 1.05)
            ax_ppl.set_ylim(0, PPL_MAX)
            ax_ppl.grid(True, alpha=0.3)
            ax_ppl.legend(fontsize=8)

            # Val loss (truncated)
            ax_vl.set_xlabel("Normalized intervention strength")
            ax_vl.set_ylabel("Validation loss")
            ax_vl.set_title(f"Val Loss  (truncated at {VAL_LOSS_MAX:.0f})")
            ax_vl.set_xlim(-0.03, 1.05)
            ax_vl.set_ylim(0, VAL_LOSS_MAX)
            ax_vl.grid(True, alpha=0.3)
            ax_vl.legend(fontsize=8)

            fig.suptitle(f"All Methods — {label}", fontsize=13, fontweight="bold")
            plt.tight_layout()

            out_path = out_dir / f"all_methods_{_format_safe(label)}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  → {out_path}")

            all_sweeps.append((label, sweeps))

        print()

    # ── Combined summary grid ─────────────────────────────────────────────
    if len(all_sweeps) > 1:
        plot_combined_summary(all_sweeps, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
