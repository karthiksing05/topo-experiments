#!/usr/bin/env python3
"""
interp_toxicity_nanogpt.py
==========================
Interpretability analysis of OSD toxic subspace directions and pruned neurons
in topo-nanoGPT 125M models.

Analyses
--------
1. **Logit Lens** — decode OSD directions and top-pruned neurons into vocabulary
   space to discover what toxic features they represent.
2. **Max-Activating Examples** — find token contexts that maximally excite
   the top OSD directions and the top toxic neurons.
3. **Cross-Layer Toxic Similarity** — principal-angle similarity between per-layer
   OSD subspaces, revealing whether layers share the same toxic features.
4. **OSD–Pruning Alignment** — measure overlap between the two intervention
   strategies: how much of each pruned-neuron output direction lies in the OSD
   toxic subspace.
5. **Tau Comparison** — every metric is computed for each tau and plotted to
   show the effect of topographic regularisation strength.

Outputs: JSON results + matplotlib figures → ``--output_dir``.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tiktoken
import torch

from huggingface_hub import hf_hub_download

# ── Import core functions from the eval script ───────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import importlib
_eval = importlib.import_module("src.toxicity.eval_toxicity_techniques_nanogpt")

load_gpt_checkpoint     = _eval.load_gpt_checkpoint
collect_mlp_activations = _eval.collect_mlp_activations
compute_neuron_t_stats  = _eval.compute_neuron_t_stats
_compute_osd_bases      = _eval._compute_osd_bases
load_realtoxicity_prompts = _eval.load_realtoxicity_prompts
load_toxigen_prompts    = _eval.load_toxigen_prompts

HF_REPO      = _eval.HF_REPO
ALL_TAUS     = _eval.ALL_TAUS
HF_CACHE     = _eval.HF_CACHE
_NON_TOXIC_TEXTS = _eval._NON_TOXIC_TEXTS

# ── Constants ─────────────────────────────────────────────────────────────────
TOP_K_TOKENS    = 15   # tokens to show per direction / neuron in logit lens
TOP_K_NEURONS   = 20   # most-toxic neurons to analyse per layer
TOP_K_DIRS      = 5    # OSD directions per layer to decode
TOP_K_EXAMPLES  = 8    # max-activating examples to collect
CONTEXT_WINDOW  = 6    # tokens of context around max-activating position


# ═════════════════════════════════════════════════════════════════════════════
# 1. LOGIT LENS
# ═════════════════════════════════════════════════════════════════════════════
def logit_lens_osd(
    model,
    osd_bases: dict[int, np.ndarray],
    tokenizer,
    device: torch.device,
    top_k_dirs: int = TOP_K_DIRS,
    top_k_tokens: int = TOP_K_TOKENS,
) -> dict[int, list[dict]]:
    """Project each OSD basis vector through c_proj → ln_f → lm_head to decode
    into vocabulary space.  Returns per-layer list of dicts with promoted /
    suppressed token rankings."""
    W_unembed = model.lm_head.weight.detach().float()          # (V, n_embd)
    ln_f_w    = model.transformer.ln_f.weight.detach().float()  # (n_embd,)
    blocks    = list(model.transformer.h)
    results: dict[int, list[dict]] = {}

    for li, block in enumerate(blocks):
        W_proj = block.mlp.c_proj.weight.detach().float()  # (n_embd, 4*n_embd)
        U = osd_bases.get(li)
        if U is None:
            continue
        n_dirs = min(top_k_dirs, U.shape[1])
        layer_results = []
        for d in range(n_dirs):
            u = torch.from_numpy(U[:, d]).to(device).float()   # (4*n_embd,)
            resid_dir = W_proj @ u                               # (n_embd,)
            resid_dir = resid_dir * ln_f_w                       # apply LN gain
            logit_contrib = W_unembed @ resid_dir                # (V,)

            vals, idxs = logit_contrib.topk(top_k_tokens)
            promoted = [(tokenizer.decode([i.item()]), v.item())
                        for i, v in zip(idxs, vals)]

            vals_b, idxs_b = logit_contrib.topk(top_k_tokens, largest=False)
            suppressed = [(tokenizer.decode([i.item()]), v.item())
                          for i, v in zip(idxs_b, vals_b)]

            layer_results.append(dict(
                direction=d,
                promoted=promoted,
                suppressed=suppressed,
            ))
        results[li] = layer_results
    return results


def logit_lens_neurons(
    model,
    t_stats: dict[int, np.ndarray],
    tokenizer,
    device: torch.device,
    top_k_neurons: int = TOP_K_NEURONS,
    top_k_tokens: int = TOP_K_TOKENS,
) -> dict[int, list[dict]]:
    """For the highest-t-stat neurons per layer, project their c_proj column
    through the unembedding to decode what tokens they promote."""
    W_unembed = model.lm_head.weight.detach().float()
    ln_f_w    = model.transformer.ln_f.weight.detach().float()
    blocks    = list(model.transformer.h)
    results: dict[int, list[dict]] = {}

    for li, block in enumerate(blocks):
        W_proj = block.mlp.c_proj.weight.detach().float()  # (n_embd, 4*n_embd)
        ts = t_stats[li]
        top_idx = np.argsort(ts)[::-1][:top_k_neurons]
        layer_results = []
        for j in top_idx:
            col_j = W_proj[:, int(j)]                    # (n_embd,)
            col_j = col_j * ln_f_w
            logit_contrib = W_unembed @ col_j            # (V,)
            vals, idxs = logit_contrib.topk(top_k_tokens)
            promoted = [(tokenizer.decode([i.item()]), v.item())
                        for i, v in zip(idxs, vals)]
            layer_results.append(dict(
                neuron=int(j),
                t_stat=float(ts[j]),
                promoted=promoted,
            ))
        results[li] = layer_results
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 2. MAX-ACTIVATING EXAMPLES
# ═════════════════════════════════════════════════════════════════════════════
def _collect_acts_with_positions(
    model, texts: list[str], tokenizer, device, max_tokens: int = 128,
) -> dict[int, tuple[np.ndarray, list[list[int]]]]:
    """Collect MLP c_fc outputs plus per-text token ids so we can recover
    context strings.  Returns dict[layer → (acts, token_ids_list)]."""
    blocks = list(model.transformer.h)
    n_layers = len(blocks)
    buffers: dict[int, list[np.ndarray]] = {i: [] for i in range(n_layers)}
    all_tok_ids: list[list[int]] = []
    handles = []

    def _make_hook(li):
        def hook(_mod, _inp, out):
            buffers[li].append(out.detach().cpu().float().numpy().reshape(-1, out.shape[-1]))
        return hook

    for li, blk in enumerate(blocks):
        handles.append(blk.mlp.c_fc.register_forward_hook(_make_hook(li)))

    with torch.no_grad():
        for text in texts:
            for li in range(n_layers):
                buffers[li].clear()
            tok_ids = tokenizer.encode_ordinary(text)[:max_tokens]
            all_tok_ids.append(tok_ids)
            inp = torch.tensor([tok_ids], device=device)
            model(inp)

    for h in handles:
        h.remove()

    # Re-collect with per-text separation
    buffers2: dict[int, list[np.ndarray]] = {i: [] for i in range(n_layers)}
    handles2 = []
    for li, blk in enumerate(blocks):
        handles2.append(blk.mlp.c_fc.register_forward_hook(_make_hook(li)))

    per_text_acts: dict[int, list[np.ndarray]] = {i: [] for i in range(n_layers)}
    with torch.no_grad():
        for text_idx, text in enumerate(texts):
            for li in range(n_layers):
                buffers[li].clear()
            tok_ids = tokenizer.encode_ordinary(text)[:max_tokens]
            inp = torch.tensor([tok_ids], device=device)
            model(inp)
            for li in range(n_layers):
                if buffers[li]:
                    per_text_acts[li].append(buffers[li][0])

    for h in handles2:
        h.remove()

    return per_text_acts, all_tok_ids


def max_activating_examples_osd(
    model, osd_bases: dict[int, np.ndarray],
    texts: list[str], tokenizer, device,
    top_k_dirs: int = 3, top_k_ex: int = TOP_K_EXAMPLES,
    context_window: int = CONTEXT_WINDOW,
) -> dict[int, list[dict]]:
    """For top OSD directions, find which text+position produces the largest
    projection and return context strings."""
    per_text_acts, all_tok_ids = _collect_acts_with_positions(
        model, texts, tokenizer, device)
    blocks = list(model.transformer.h)
    results: dict[int, list[dict]] = {}

    for li in range(len(blocks)):
        U = osd_bases.get(li)
        if U is None:
            continue
        n_dirs = min(top_k_dirs, U.shape[1])
        layer_results = []
        for d in range(n_dirs):
            u = U[:, d]  # (4*n_embd,)
            # Gather (text_idx, pos, projection_value)
            candidates = []
            for ti, act_arr in enumerate(per_text_acts[li]):
                proj = act_arr @ u  # (n_tokens,)
                for pos in range(len(proj)):
                    candidates.append((abs(float(proj[pos])), float(proj[pos]),
                                       ti, pos))
            candidates.sort(key=lambda x: x[0], reverse=True)
            examples = []
            for _, proj_val, ti, pos in candidates[:top_k_ex]:
                toks = all_tok_ids[ti]
                start = max(0, pos - context_window)
                end = min(len(toks), pos + context_window + 1)
                context = tokenizer.decode(toks[start:end])
                target_tok = tokenizer.decode([toks[pos]]) if pos < len(toks) else "?"
                examples.append(dict(
                    text_idx=ti, position=pos, projection=proj_val,
                    target_token=target_tok, context=context,
                ))
            layer_results.append(dict(direction=d, examples=examples))
        results[li] = layer_results
    return results


def max_activating_examples_neurons(
    model, t_stats: dict[int, np.ndarray],
    texts: list[str], tokenizer, device,
    top_k_neurons: int = 10, top_k_ex: int = TOP_K_EXAMPLES,
    context_window: int = CONTEXT_WINDOW,
) -> dict[int, list[dict]]:
    """For highest-t-stat neurons, find max-activating token contexts."""
    per_text_acts, all_tok_ids = _collect_acts_with_positions(
        model, texts, tokenizer, device)
    blocks = list(model.transformer.h)
    results: dict[int, list[dict]] = {}

    for li in range(len(blocks)):
        ts = t_stats[li]
        top_idx = np.argsort(ts)[::-1][:top_k_neurons]
        layer_results = []
        for j in top_idx:
            j = int(j)
            candidates = []
            for ti, act_arr in enumerate(per_text_acts[li]):
                vals = act_arr[:, j]
                for pos in range(len(vals)):
                    candidates.append((float(vals[pos]), ti, pos))
            candidates.sort(key=lambda x: x[0], reverse=True)
            examples = []
            for val, ti, pos in candidates[:top_k_ex]:
                toks = all_tok_ids[ti]
                start = max(0, pos - context_window)
                end = min(len(toks), pos + context_window + 1)
                context = tokenizer.decode(toks[start:end])
                target_tok = tokenizer.decode([toks[pos]]) if pos < len(toks) else "?"
                examples.append(dict(
                    text_idx=ti, position=pos, activation=val,
                    target_token=target_tok, context=context,
                ))
            layer_results.append(dict(
                neuron=j, t_stat=float(ts[j]), examples=examples))
        results[li] = layer_results
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 3. CROSS-LAYER OSD SIMILARITY
# ═════════════════════════════════════════════════════════════════════════════
def cross_layer_osd_similarity(
    osd_bases: dict[int, np.ndarray],
) -> np.ndarray:
    """Compute pairwise subspace similarity between layers using the mean
    squared cosine of principal angles (Grassmann affinity)."""
    layers = sorted(osd_bases.keys())
    n = len(layers)
    sim = np.eye(n)
    for i in range(n):
        Ui = osd_bases[layers[i]]  # (D, ki)
        for j in range(i + 1, n):
            Uj = osd_bases[layers[j]]
            # principal angles via SVD of Ui.T @ Uj
            M = Ui.T @ Uj
            svals = np.linalg.svd(M, compute_uv=False)
            # mean squared cosine of principal angles
            affinity = float(np.mean(svals ** 2))
            sim[i, j] = sim[j, i] = affinity
    return sim


# ═════════════════════════════════════════════════════════════════════════════
# 4. OSD–PRUNING ALIGNMENT
# ═════════════════════════════════════════════════════════════════════════════
def osd_pruning_alignment(
    model,
    osd_bases: dict[int, np.ndarray],
    t_stats: dict[int, np.ndarray],
    frac: float = 0.2,
) -> dict[int, dict]:
    """Measure how much the output directions of pruned neurons lie within
    the OSD toxic subspace.

    For each layer, take the top-frac% neurons (highest t-stat), get their
    c_proj columns, project onto the OSD subspace, and report the fraction
    of squared norm captured.  High values → the two methods target the same
    features."""
    blocks = list(model.transformer.h)
    results: dict[int, dict] = {}
    for li, block in enumerate(blocks):
        U = osd_bases.get(li)
        ts = t_stats.get(li)
        if U is None or ts is None:
            continue
        W_proj = block.mlp.c_proj.weight.detach().cpu().float().numpy()  # (n_embd, 4*n_embd)
        n_neurons = len(ts)
        k = max(1, round(frac * n_neurons))
        top_idx = np.argsort(ts)[::-1][:k]

        # output directions of pruned neurons: columns of W_proj
        D = W_proj[:, top_idx]       # (n_embd, k)

        # Project each column onto OSD subspace in MLP-hidden space
        # But OSD lives in (4*n_embd) space... we need the alignment there.
        # Instead: measure how much of c_fc activation on pruned neurons
        # is captured by OSD directions.
        # Simpler: for each pruned neuron j, its unit vector e_j in MLP-hidden space.
        # Fraction of e_j in OSD subspace = ||U^T e_j||^2 (since U is orthonormal-ish)
        # Normalise U to be orthonormal
        Q, _ = np.linalg.qr(U)  # (4*n_embd, k_osd)

        captured_fracs = []
        for j in top_idx:
            ej = np.zeros(U.shape[0], dtype=np.float32)
            ej[int(j)] = 1.0
            proj_norm_sq = float(np.sum((Q.T @ ej) ** 2))
            captured_fracs.append(proj_norm_sq)

        results[li] = dict(
            n_pruned=k,
            mean_alignment=float(np.mean(captured_fracs)),
            median_alignment=float(np.median(captured_fracs)),
            max_alignment=float(np.max(captured_fracs)),
            min_alignment=float(np.min(captured_fracs)),
            per_neuron=[(int(j), float(c))
                        for j, c in zip(top_idx[:10], captured_fracs[:10])],
        )
    return results


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═════════════════════════════════════════════════════════════════════════════
def plot_logit_lens_osd(ll_results: dict, label: str, output_dir: Path):
    """Grid of top-promoted tokens per OSD direction per layer."""
    layers = sorted(ll_results.keys())
    n_layers = len(layers)
    n_dirs = max(len(ll_results[l]) for l in layers) if layers else 0
    if n_dirs == 0:
        return

    fig, axes = plt.subplots(n_dirs, 1, figsize=(14, 3.5 * n_dirs), squeeze=False)
    fig.suptitle(f"OSD Logit Lens — {label}", fontsize=14, y=1.01)

    for d in range(n_dirs):
        ax = axes[d, 0]
        tokens_grid = []
        scores_grid = []
        layer_labels = []
        for li in layers:
            if d < len(ll_results[li]):
                entry = ll_results[li][d]
                toks = [t for t, _ in entry["promoted"][:10]]
                vals = [v for _, v in entry["promoted"][:10]]
                tokens_grid.append(toks)
                scores_grid.append(vals)
                layer_labels.append(f"L{li}")

        if not tokens_grid:
            ax.set_visible(False)
            continue

        n_tok = max(len(row) for row in tokens_grid)
        data = np.zeros((len(tokens_grid), n_tok))
        annot = np.full((len(tokens_grid), n_tok), "", dtype=object)
        for r, (toks, vals) in enumerate(zip(tokens_grid, scores_grid)):
            for c, (t, v) in enumerate(zip(toks, vals)):
                data[r, c] = v
                annot[r, c] = repr(t).strip("'")

        im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                if annot[r, c]:
                    ax.text(c, r, annot[r, c], ha="center", va="center",
                            fontsize=6, color="black")
        ax.set_yticks(range(len(layer_labels)))
        ax.set_yticklabels(layer_labels, fontsize=8)
        ax.set_xticks([])
        ax.set_title(f"OSD direction {d} — top promoted tokens", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.6, label="logit contribution")

    fig.tight_layout()
    fig.savefig(output_dir / f"logit_lens_osd__{label}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_logit_lens_neurons(ll_results: dict, label: str, output_dir: Path):
    """Bar chart of top promoted tokens for the top-3 neurons per layer."""
    layers = sorted(ll_results.keys())
    show_per_layer = 3
    n_layers = len(layers)
    fig, axes = plt.subplots(n_layers, show_per_layer,
                             figsize=(5 * show_per_layer, 2.4 * n_layers),
                             squeeze=False)
    fig.suptitle(f"Neuron Logit Lens — {label}", fontsize=14, y=1.01)

    for row, li in enumerate(layers):
        entries = ll_results[li][:show_per_layer]
        for col, entry in enumerate(entries):
            ax = axes[row, col]
            promoted = entry["promoted"][:10]
            toks = [repr(t).strip("'") for t, _ in promoted]
            vals = [v for _, v in promoted]
            ax.barh(range(len(toks)), vals, color="salmon")
            ax.set_yticks(range(len(toks)))
            ax.set_yticklabels(toks, fontsize=7)
            ax.invert_yaxis()
            ax.set_title(f"L{li} n{entry['neuron']} (t={entry['t_stat']:.1f})",
                         fontsize=8)
        for col in range(len(entries), show_per_layer):
            axes[row, col].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / f"logit_lens_neurons__{label}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_cross_layer_sim(sim: np.ndarray, label: str, output_dir: Path):
    """Heatmap of pairwise OSD subspace similarity."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
    n = sim.shape[0]
    ax.set_xticks(range(n)); ax.set_xticklabels([f"L{i}" for i in range(n)], fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels([f"L{i}" for i in range(n)], fontsize=8)
    ax.set_title(f"Cross-Layer OSD Subspace Similarity — {label}", fontsize=11)
    plt.colorbar(im, ax=ax, label="Grassmann affinity")
    fig.tight_layout()
    fig.savefig(output_dir / f"cross_layer_osd_sim__{label}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_osd_pruning_alignment(
    alignment_per_tau: dict[str, dict[int, dict]],
    output_dir: Path,
):
    """Bar chart of mean OSD–pruning alignment per layer, across taus."""
    fig, ax = plt.subplots(figsize=(10, 5))
    tau_labels = sorted(alignment_per_tau.keys())
    n_taus = len(tau_labels)
    all_layers = set()
    for a in alignment_per_tau.values():
        all_layers.update(a.keys())
    layers = sorted(all_layers)
    n_layers = len(layers)
    if n_layers == 0:
        plt.close(fig)
        return

    bar_w = 0.8 / max(n_taus, 1)
    for ti, tl in enumerate(tau_labels):
        means = [alignment_per_tau[tl].get(li, {}).get("mean_alignment", 0)
                 for li in layers]
        x = np.arange(n_layers) + ti * bar_w
        ax.bar(x, means, bar_w, label=tl, alpha=0.85)

    ax.set_xticks(np.arange(n_layers) + bar_w * (n_taus - 1) / 2)
    ax.set_xticklabels([f"L{li}" for li in layers], fontsize=8)
    ax.set_ylabel("Mean alignment (frac of neuron in OSD subspace)")
    ax.set_title("OSD–Pruning Alignment by Layer and Tau")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "osd_pruning_alignment_all_taus.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_tau_comparison_summary(
    tau_summaries: dict[str, dict],
    output_dir: Path,
):
    """Multi-panel figure: how toxic neuron count, OSD variance, and alignment
    change across taus."""
    taus = sorted(tau_summaries.keys(), key=lambda t: float(t.replace("tau", "")))
    if not taus:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: mean t-stat of top-20% neurons per layer
    ax = axes[0]
    for tl in taus:
        s = tau_summaries[tl]
        layers = sorted(s["mean_top_tstat_per_layer"].keys(), key=int)
        vals = [s["mean_top_tstat_per_layer"][l] for l in layers]
        ax.plot(range(len(layers)), vals, "o-", label=tl, markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean t-stat (top 20% neurons)")
    ax.set_title("Toxic Neuron Strength")
    ax.legend(fontsize=7)

    # Panel 2: total OSD variance per layer (sum of squared singular values)
    ax = axes[1]
    for tl in taus:
        s = tau_summaries[tl]
        layers = sorted(s["osd_variance_per_layer"].keys(), key=int)
        vals = [s["osd_variance_per_layer"][l] for l in layers]
        ax.plot(range(len(layers)), vals, "o-", label=tl, markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Total OSD variance (Σ σ²)")
    ax.set_title("OSD Toxic Subspace Strength")
    ax.legend(fontsize=7)

    # Panel 3: mean alignment per tau
    ax = axes[2]
    mean_aligns = []
    for tl in taus:
        s = tau_summaries[tl]
        vals = list(s["mean_alignment_per_layer"].values())
        mean_aligns.append(np.mean(vals) if vals else 0)
    ax.bar(range(len(taus)), mean_aligns, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels(taus, fontsize=8, rotation=30)
    ax.set_ylabel("Mean OSD–Pruning alignment")
    ax.set_title("Alignment Across Taus")

    fig.suptitle("Tau Comparison — Interpretability Summary", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "tau_comparison_summary.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_max_act_examples(
    examples_results: dict[int, list[dict]],
    kind: str,
    label: str,
    output_dir: Path,
):
    """Save a text-based summary figure of max-activating examples."""
    lines = [f"Max-Activating Examples ({kind}) — {label}\n" + "=" * 60 + "\n"]
    for li in sorted(examples_results.keys()):
        entries = examples_results[li][:3]  # top 3 dirs/neurons
        for entry in entries:
            if kind == "OSD":
                header = f"Layer {li}, Direction {entry['direction']}"
            else:
                header = f"Layer {li}, Neuron {entry['neuron']} (t={entry['t_stat']:.2f})"
            lines.append(f"\n{header}")
            lines.append("-" * len(header))
            for ex in entry["examples"][:5]:
                tok = ex.get("target_token", "?")
                val_key = "projection" if "projection" in ex else "activation"
                val = ex[val_key]
                ctx = ex["context"].replace("\n", "\\n")
                lines.append(f"  [{tok:>12s}] val={val:+.3f}  ...{ctx}...")
    text = "\n".join(lines)
    path = output_dir / f"max_act_examples_{kind.lower()}__{label}.txt"
    path.write_text(text)
    print(f"  Saved {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description="Interpretability analysis for OSD and pruning in topo-nanoGPT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--taus", type=str,
                   default=",".join(str(t) for t in ALL_TAUS),
                   help="Comma-separated tau values")
    p.add_argument("--n_prompts", type=int, default=200,
                   help="Toxic prompts to load per dataset")
    p.add_argument("--n_selectivity_tokens", type=int, default=4096,
                   help="Token budget for activation collection")
    p.add_argument("--n_osd_components", type=int, default=32)
    p.add_argument("--n_clean_components", type=int, default=32)
    p.add_argument("--alignment_frac", type=float, default=0.2,
                   help="Pruning fraction for OSD-pruning alignment analysis")
    p.add_argument("--dataset", choices=["rtp", "toxigen", "both"], default="rtp",
                   help="Which dataset to use for activations")
    p.add_argument("--no_logit_lens", action="store_true",
                   help="Skip logit lens analysis")
    p.add_argument("--no_max_act", action="store_true",
                   help="Skip max-activating examples")
    p.add_argument("--no_cross_layer", action="store_true",
                   help="Skip cross-layer OSD similarity")
    p.add_argument("--no_alignment", action="store_true",
                   help="Skip OSD-pruning alignment")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str,
                   default="outputs/interp_toxicity_nanogpt")
    args = p.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    taus = [float(t) for t in args.taus.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Taus: {taus}")
    print(f"Output: {output_dir}")

    tokenizer = tiktoken.get_encoding("gpt2")

    # ── Load prompts ──────────────────────────────────────────────────────
    print("\nLoading prompts…")
    toxic_prompts: list[str] = []
    if args.dataset in ("rtp", "both"):
        rtp = load_realtoxicity_prompts(args.n_prompts)
        toxic_prompts.extend(rtp)
        print(f"  RTP: {len(rtp)} prompts")
    if args.dataset in ("toxigen", "both"):
        tg = load_toxigen_prompts(args.n_prompts)
        toxic_prompts.extend(tg)
        print(f"  ToxiGen: {len(tg)} prompts")
    nontoxic_texts = list(_NON_TOXIC_TEXTS) * 4
    print(f"  Clean: {len(nontoxic_texts)} prompts")

    max_toks = max(1, args.n_selectivity_tokens // max(len(toxic_prompts), 1))

    # ── Per-tau analysis ──────────────────────────────────────────────────
    all_results: dict[str, dict] = {}
    alignment_per_tau: dict[str, dict] = {}
    tau_summaries: dict[str, dict] = {}

    for tau in taus:
        label = f"tau{tau}".replace(".", "_")
        print(f"\n{'='*60}")
        print(f"  τ = {tau}  ({label})")
        print(f"{'='*60}")

        # Download + load
        filename = f"tau_{tau}.pt"
        ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=filename,
                                    cache_dir=str(HF_CACHE))
        model = load_gpt_checkpoint(ckpt_path, device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Model loaded: {n_params:.1f}M params")

        # Collect activations
        print(f"  Collecting MLP activations…")
        toxic_acts = collect_mlp_activations(
            model, toxic_prompts, tokenizer, device, max_toks)
        nontoxic_acts = collect_mlp_activations(
            model, nontoxic_texts, tokenizer, device, max_toks)
        print(f"  Toxic tokens: {toxic_acts[0].shape[0]}, "
              f"Clean tokens: {nontoxic_acts[0].shape[0]}")

        # Compute heuristics
        print(f"  Computing neuron t-stats…")
        t_stats = compute_neuron_t_stats(toxic_acts, nontoxic_acts)

        print(f"  Computing OSD bases…")
        osd_bases, osd_svals = _compute_osd_bases(
            toxic_acts, nontoxic_acts,
            n_toxic=args.n_osd_components, n_clean=args.n_clean_components)

        tau_result: dict = {"tau": tau, "label": label}

        # ── Analysis 1: Logit Lens ────────────────────────────────────────
        if not args.no_logit_lens:
            print(f"  [1/4] Logit lens on OSD directions…")
            ll_osd = logit_lens_osd(model, osd_bases, tokenizer, device)
            tau_result["logit_lens_osd"] = {
                str(k): v for k, v in ll_osd.items()}
            plot_logit_lens_osd(ll_osd, label, output_dir)

            print(f"  [1/4] Logit lens on pruned neurons…")
            ll_neurons = logit_lens_neurons(model, t_stats, tokenizer, device)
            tau_result["logit_lens_neurons"] = {
                str(k): v for k, v in ll_neurons.items()}
            plot_logit_lens_neurons(ll_neurons, label, output_dir)

        # ── Analysis 2: Max-Activating Examples ───────────────────────────
        if not args.no_max_act:
            # Use a subset of prompts for speed
            example_texts = toxic_prompts[:50] + nontoxic_texts[:10]
            print(f"  [2/4] Max-activating examples (OSD)…")
            max_osd = max_activating_examples_osd(
                model, osd_bases, example_texts, tokenizer, device)
            tau_result["max_act_osd"] = {str(k): v for k, v in max_osd.items()}
            plot_max_act_examples(max_osd, "OSD", label, output_dir)

            print(f"  [2/4] Max-activating examples (neurons)…")
            max_neurons = max_activating_examples_neurons(
                model, t_stats, example_texts, tokenizer, device)
            tau_result["max_act_neurons"] = {
                str(k): v for k, v in max_neurons.items()}
            plot_max_act_examples(max_neurons, "Neuron", label, output_dir)

        # ── Analysis 3: Cross-Layer Similarity ────────────────────────────
        if not args.no_cross_layer:
            print(f"  [3/4] Cross-layer OSD similarity…")
            sim = cross_layer_osd_similarity(osd_bases)
            tau_result["cross_layer_sim"] = sim.tolist()
            plot_cross_layer_sim(sim, label, output_dir)

        # ── Analysis 4: OSD–Pruning Alignment ────────────────────────────
        if not args.no_alignment:
            print(f"  [4/4] OSD–Pruning alignment…")
            alignment = osd_pruning_alignment(
                model, osd_bases, t_stats, frac=args.alignment_frac)
            tau_result["osd_pruning_alignment"] = {
                str(k): v for k, v in alignment.items()}
            alignment_per_tau[label] = alignment

        # ── Tau summary stats ─────────────────────────────────────────────
        summary: dict = {
            "mean_top_tstat_per_layer": {},
            "osd_variance_per_layer": {},
            "mean_alignment_per_layer": {},
        }
        for li in sorted(t_stats.keys()):
            ts = t_stats[li]
            k20 = max(1, round(0.2 * len(ts)))
            top20 = np.sort(ts)[::-1][:k20]
            summary["mean_top_tstat_per_layer"][str(li)] = float(np.mean(top20))
        for li in sorted(osd_svals.keys()):
            sv = osd_svals[li]
            summary["osd_variance_per_layer"][str(li)] = float(np.sum(sv ** 2))
        if not args.no_alignment and label in alignment_per_tau:
            for li, a in alignment_per_tau[label].items():
                summary["mean_alignment_per_layer"][str(li)] = a["mean_alignment"]
        tau_summaries[label] = summary
        tau_result["summary"] = summary

        all_results[label] = tau_result

        # free GPU memory
        del model
        torch.cuda.empty_cache()

    # ── Cross-tau plots ───────────────────────────────────────────────────
    if len(taus) > 1:
        print(f"\nGenerating cross-tau comparison plots…")
        if not args.no_alignment:
            plot_osd_pruning_alignment(alignment_per_tau, output_dir)
        plot_tau_comparison_summary(tau_summaries, output_dir)

    # ── Save JSON ─────────────────────────────────────────────────────────
    out_json = output_dir / "interp_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_json}")
    print("Done.")


if __name__ == "__main__":
    main()
