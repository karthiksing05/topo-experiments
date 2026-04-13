#!/usr/bin/env python3
"""
eval_toxicity_strategies_nanogpt_450m.py
=========================================
450M variant of eval_toxicity_strategies_nanogpt.py.

Differences from the 125M version:
  - Local checkpoints (CKPT_ROOT) instead of HuggingFace
  - model.blocks[i] instead of model.transformer.h[i]
  - Two-file loading: config JSON + step checkpoint
  - Integer taus: [0, 30722, 307226]
"""
from __future__ import annotations

import argparse, copy, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import shared infrastructure from the 450M techniques eval ────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import importlib as _il
_te = _il.import_module("src.toxicity.eval_toxicity_techniques_nanogpt_450m")

# model / data
GPT                        = _te.GPT
load_gpt_checkpoint        = _te.load_gpt_checkpoint
load_realtoxicity_prompts  = _te.load_realtoxicity_prompts
load_toxigen_prompts       = _te.load_toxigen_prompts
_NON_TOXIC_TEXTS           = _te._NON_TOXIC_TEXTS
_load_owt_reference        = _te._load_owt_reference
CKPT_ROOT                  = _te.CKPT_ROOT
ALL_TAUS                   = _te.ALL_TAUS
FINAL_STEP                 = _te.FINAL_STEP

# activation collection & heuristics
collect_mlp_activations    = _te.collect_mlp_activations
compute_neuron_t_stats     = _te.compute_neuron_t_stats
_compute_osd_bases         = _te._compute_osd_bases
_collect_residual_means    = _te._collect_residual_means

# evaluation / scoring
evaluate_model_full        = _te.evaluate_model_full
generate_continuations     = _te.generate_continuations
compute_perplexity         = _te.compute_perplexity
score_texts_detoxify       = _te.score_texts_detoxify
aggregate_scores           = _te.aggregate_scores

# weight helpers
_snapshot_c_proj           = _te._snapshot_c_proj
_restore_c_proj            = _te._restore_c_proj
_subdir                    = _te._subdir
_prepend_baseline          = _te._prepend_baseline

# constants
_DATASET_KEYS   = _te._DATASET_KEYS
_DATASET_LABELS = _te._DATASET_LABELS

# ── Strategy labels (for plots) ──────────────────────────────────────────────
STRATEGY_LABELS: dict[str, str] = {
    "eigenshift":        "EigenShift",
    "self_debiasing":    "Self-Debiasing",
    "chars_lite":        "CHaRS-lite",
    "vocab_shifting":    "Vocab Shifting",
    "pct_osd":           "PCT (thresholded OSD)",
}

STRATEGY_FAMILIES: dict[str, str] = {
    "eigenshift":        "mechanistic",
    "self_debiasing":    "decoding",
    "chars_lite":        "mechanistic",
    "vocab_shifting":    "decoding",
    "pct_osd":           "mechanistic",
}


# ═════════════════════════════════════════════════════════════════════════════
# 1.  EIGENSHIFT  (450M: model.blocks[i])
# ═════════════════════════════════════════════════════════════════════════════

def _eigenshift_score_eigenvectors(
    model: GPT,
    toxic_acts: dict[int, np.ndarray],
    nontoxic_acts: dict[int, np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    W = model.lm_head.weight.detach().float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    Vt_np = Vt.cpu().numpy()
    k = Vt_np.shape[0]
    n_layers = len(model.blocks)
    scores = np.zeros(k, dtype=np.float64)

    for li in range(n_layers):
        W_proj = model.blocks[li].mlp.c_proj.weight.detach().cpu().float().numpy()
        t_mean = toxic_acts[li].mean(axis=0)
        c_mean = nontoxic_acts[li].mean(axis=0)
        t_resid = W_proj @ t_mean
        c_resid = W_proj @ c_mean
        diff = t_resid - c_resid
        for i in range(k):
            scores[i] += abs(float(Vt_np[i] @ diff))

    return U.cpu().numpy(), S.cpu().numpy(), Vt_np, scores


def _apply_eigenshift(model, U, S, Vt, toxicity_scores, frac, device):
    k = len(S)
    n_suppress = max(1, round(frac * k))
    order = np.argsort(toxicity_scores)[::-1]
    suppress_idx = set(order[:n_suppress].tolist())
    S_new = S.copy()
    for idx in suppress_idx:
        S_new[idx] = 0.0
    W_new = torch.tensor(U, dtype=torch.float32, device=device) \
          @ torch.diag(torch.tensor(S_new, dtype=torch.float32, device=device)) \
          @ torch.tensor(Vt, dtype=torch.float32, device=device)
    with torch.no_grad():
        model.lm_head.weight.copy_(W_new)


def _restore_lm_head(model, lm_head_snap):
    with torch.no_grad():
        model.lm_head.weight.copy_(lm_head_snap)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  SELF-DEBIASING
# ═════════════════════════════════════════════════════════════════════════════

_TOXIC_PREFIX = "The following text contains hate speech and offensive language: "

def generate_self_debiased(
    model, tokenizer, prompts, device,
    alpha=0.5, max_new_tokens=200, temperature=1.0, top_k=50,
):
    continuations = []
    for prompt in prompts:
        tok_normal = tokenizer.encode_ordinary(prompt)
        tok_biased = tokenizer.encode_ordinary(_TOXIC_PREFIX + prompt)
        idx_n = torch.tensor([tok_normal], device=device)
        idx_b = torch.tensor([tok_biased], device=device)
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits_n = model(idx_n[:, -model.block_size:])[:, -1, :]
                logits_b = model(idx_b[:, -model.block_size:])[:, -1, :]
            logits_adj = logits_n - alpha * (logits_b - logits_n)
            if temperature != 1.0:
                logits_adj = logits_adj / temperature
            if top_k is not None:
                v, _ = torch.topk(logits_adj, min(top_k, logits_adj.size(-1)))
                logits_adj[logits_adj < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits_adj, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx_n = torch.cat([idx_n, next_tok], dim=1)
            idx_b = torch.cat([idx_b, next_tok], dim=1)
        out_ids = idx_n[0, len(tok_normal):].tolist()
        continuations.append(tokenizer.decode(out_ids))
    return continuations


def evaluate_self_debiased(
    model, tokenizer, prompts, detox_model, device, owt_ref_text,
    alpha=0.5, max_new_tokens=200, temperature=1.0, top_k=50,
    llamaguard_scorer=None,
):
    conts = generate_self_debiased(
        model, tokenizer, prompts, device, alpha,
        max_new_tokens, temperature, top_k)
    raw_det = score_texts_detoxify(conts, detox_model)
    agg_det = aggregate_scores(raw_det)
    agg_lg = None
    if llamaguard_scorer is not None:
        rows = llamaguard_scorer.score_texts(conts)
        if rows:
            from src.toxicity.eval_toxicity_techniques_nanogpt_450m import aggregate_llamaguard
            agg_lg = aggregate_llamaguard(rows)
    ppl, val_loss = compute_perplexity(model, tokenizer, owt_ref_text, device)
    return {
        "n_prompts": len(prompts),
        "n_completions": len(conts),
        "detoxify": agg_det,
        "llamaguard": agg_lg,
        "perplexity": ppl,
        "val_loss": val_loss,
        "per_completion_toxicity": raw_det.get("toxicity", []),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3.  CHaRS-LITE  (450M: model.blocks[i])
# ═════════════════════════════════════════════════════════════════════════════

def _compute_chars_directions(toxic_acts, nontoxic_acts, n_clusters=5):
    from sklearn.cluster import MiniBatchKMeans
    result = {}
    for li in toxic_acts:
        T = toxic_acts[li]
        C = nontoxic_acts[li]
        c_mean = C.mean(axis=0)
        n_clust = min(n_clusters, T.shape[0])
        if n_clust < 2:
            continue
        km = MiniBatchKMeans(n_clusters=n_clust, random_state=42,
                             batch_size=min(1024, T.shape[0]))
        labels = km.fit_predict(T)
        centroids = km.cluster_centers_
        steering = centroids - c_mean[None, :]
        norms = np.linalg.norm(steering, axis=1, keepdims=True) + 1e-8
        steering = steering / norms
        result[li] = (centroids, steering, labels)
    return result


def _apply_chars_steering(model, chars_data, frac, snap):
    with torch.no_grad():
        for li, w in snap.items():
            model.blocks[li].mlp.c_proj.weight.copy_(w)
        for li, (centroids, steering, labels) in chars_data.items():
            if li >= len(model.blocks):
                continue
            W = model.blocks[li].mlp.c_proj.weight  # (model_dim, D)
            n_clusters = centroids.shape[0]
            n_total = max(len(labels), 1)
            for cl in range(n_clusters):
                cluster_weight = float((labels == cl).sum()) / n_total
                if cluster_weight < 1e-6:
                    continue
                s = torch.tensor(steering[cl], dtype=W.dtype, device=W.device)
                Ws = W @ s  # (model_dim,)
                W.sub_(frac * cluster_weight * Ws.unsqueeze(1) * s.unsqueeze(0))


# ═════════════════════════════════════════════════════════════════════════════
# 4.  VOCAB SHIFTING
# ═════════════════════════════════════════════════════════════════════════════

def _compute_toxic_token_scores(model, toxic_prompts, clean_prompts,
                                tokenizer, device, max_tokens=64):
    vocab_size = model.lm_head.weight.shape[0]
    toxic_logprob_sum = np.zeros(vocab_size, dtype=np.float64)
    clean_logprob_sum = np.zeros(vocab_size, dtype=np.float64)
    n_tox = n_cln = 0
    with torch.no_grad():
        for text in toxic_prompts[:50]:
            toks = tokenizer.encode_ordinary(text)[:max_tokens]
            if not toks:
                continue
            inp = torch.tensor([toks], device=device)
            logits = model(inp)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            toxic_logprob_sum += log_probs[0].cpu().numpy()
            n_tox += 1
        for text in clean_prompts[:50]:
            toks = tokenizer.encode_ordinary(text)[:max_tokens]
            if not toks:
                continue
            inp = torch.tensor([toks], device=device)
            logits = model(inp)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            clean_logprob_sum += log_probs[0].cpu().numpy()
            n_cln += 1
    if n_tox > 0:
        toxic_logprob_sum /= n_tox
    if n_cln > 0:
        clean_logprob_sum /= n_cln
    return (toxic_logprob_sum - clean_logprob_sum).astype(np.float32)


def generate_vocab_shifted(model, tokenizer, prompts, toxic_token_scores,
                           device, alpha=0.5, max_new_tokens=200,
                           temperature=1.0, top_k=50):
    penalty = torch.tensor(toxic_token_scores, dtype=torch.float32,
                           device=device)
    continuations = []
    for prompt in prompts:
        toks = tokenizer.encode_ordinary(prompt)
        idx = torch.tensor([toks], device=device)
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = model(idx[:, -model.block_size:])[:, -1, :]
            logits = logits - alpha * penalty
            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)
        out_ids = idx[0, len(toks):].tolist()
        continuations.append(tokenizer.decode(out_ids))
    return continuations


def evaluate_vocab_shifted(model, tokenizer, prompts, toxic_token_scores,
                           detox_model, device, owt_ref_text,
                           alpha=0.5, max_new_tokens=200, temperature=1.0,
                           top_k=50, llamaguard_scorer=None):
    conts = generate_vocab_shifted(
        model, tokenizer, prompts, toxic_token_scores, device,
        alpha, max_new_tokens, temperature, top_k)
    raw_det = score_texts_detoxify(conts, detox_model)
    agg_det = aggregate_scores(raw_det)
    agg_lg = None
    if llamaguard_scorer is not None:
        rows = llamaguard_scorer.score_texts(conts)
        if rows:
            from src.toxicity.eval_toxicity_techniques_nanogpt_450m import aggregate_llamaguard
            agg_lg = aggregate_llamaguard(rows)
    ppl, val_loss = compute_perplexity(model, tokenizer, owt_ref_text, device)
    return {
        "n_prompts": len(prompts),
        "n_completions": len(conts),
        "detoxify": agg_det,
        "llamaguard": agg_lg,
        "perplexity": ppl,
        "val_loss": val_loss,
        "per_completion_toxicity": raw_det.get("toxicity", []),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5.  PCT — Principal Component Thresholding  (450M: model.blocks[i])
# ═════════════════════════════════════════════════════════════════════════════

def _compute_pct_bases(toxic_acts, nontoxic_acts, n_toxic=32, n_clean=32,
                       eigenvalue_threshold=0.1):
    osd_bases, osd_svals = _compute_osd_bases(
        toxic_acts, nontoxic_acts, n_toxic=n_toxic, n_clean=n_clean)
    pct_bases, pct_svals = {}, {}
    for li in osd_bases:
        sv = osd_svals[li]
        U = osd_bases[li]
        if len(sv) == 0:
            continue
        threshold = eigenvalue_threshold * sv[0]
        keep = sv >= threshold
        if not keep.any():
            continue
        pct_bases[li] = U[:, keep]
        pct_svals[li] = sv[keep]
    return pct_bases, pct_svals


def _apply_pct_osd(model, pct_bases, pct_svals, frac, snap):
    all_pcs = []
    for li, sv in pct_svals.items():
        for j, s in enumerate(sv):
            all_pcs.append((float(s), li, j))
    if not all_pcs:
        return
    all_pcs.sort(key=lambda x: x[0], reverse=True)
    n_remove = max(1, round(frac * len(all_pcs)))
    to_remove = all_pcs[:n_remove]
    by_layer = defaultdict(list)
    for _, li, j in to_remove:
        by_layer[li].append(j)
    with torch.no_grad():
        for li, w in snap.items():
            model.blocks[li].mlp.c_proj.weight.copy_(w)
        for li, pc_indices in by_layer.items():
            if li not in pct_bases or li >= len(model.blocks):
                continue
            U = pct_bases[li]
            cols = sorted(set(pc_indices))
            Uk = torch.tensor(U[:, cols], dtype=snap[li].dtype,
                              device=snap[li].device)
            W = model.blocks[li].mlp.c_proj.weight
            W.sub_(W @ Uk @ Uk.T)


# ═════════════════════════════════════════════════════════════════════════════
# SWEEP RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_strategy_sweep(
    model, tokenizer, prompts, nontoxic_texts, detox_model, device,
    baseline_result, owt_ref_text, fracs,
    n_selectivity_tokens=4096, max_new_tokens=200, temperature=1.0,
    top_k=50, n_gen=1, llamaguard_scorer=None,
    skip_eigenshift=False, skip_self_debiasing=False,
    skip_chars=False, skip_vocab_shifting=False, skip_pct=False,
    existing=None, save_fn=None,
):
    """Run all new strategies at every frac.

    *save_fn* is an optional ``(results_dict) -> None`` callback invoked after
    each method completes so that progress is persisted incrementally.
    """
    max_toks = max(1, n_selectivity_tokens // max(1, len(prompts)))
    base_ppl = baseline_result["perplexity"]
    base_vl  = baseline_result.get("val_loss", float("nan"))

    def _eval_and_record():
        res = evaluate_model_full(
            model=model, tokenizer=tokenizer, prompts=prompts,
            detox_model=detox_model, device=device, owt_ref_text=owt_ref_text,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, n_gen=n_gen, llamaguard_scorer=llamaguard_scorer,
        )
        ppl_ratio = res["perplexity"] / max(base_ppl, 1e-6)
        vl_ratio  = res["val_loss"] / max(base_vl, 1e-6) \
            if not math.isnan(res["val_loss"]) else float("nan")
        return {**res, "ppl_ratio": ppl_ratio, "val_loss_ratio": vl_ratio}

    def _save_incremental():
        if save_fn is not None:
            save_fn(results)

    results = existing if existing else {}
    if "fracs" not in results:
        results["fracs"] = fracs
    if "baseline" not in results:
        results["baseline"] = baseline_result

    # Save baseline immediately
    _save_incremental()

    need_acts = (not skip_eigenshift) or (not skip_chars) or (not skip_pct)
    toxic_acts = nontoxic_acts_collected = None
    if need_acts:
        print(f"    Collecting MLP activations ({max_toks} tok/text)…")
        toxic_acts = collect_mlp_activations(
            model, prompts, tokenizer, device, max_toks)
        nontoxic_acts_collected = collect_mlp_activations(
            model, nontoxic_texts, tokenizer, device, max_toks)

    snap = _snapshot_c_proj(model)
    lm_head_snap = model.lm_head.weight.data.clone()

    cuda_ok = True

    # ── 1. EigenShift ─────────────────────────────────────────────────────
    if not skip_eigenshift:
        if "eigenshift" in results and results["eigenshift"]:
            print(f"  ── eigenshift (already done, skipping) ──")
        elif not cuda_ok:
            print(f"  ── eigenshift (skipped – GPU context corrupted) ──")
            results["eigenshift"] = {}
        else:
            print(f"  ── eigenshift ──")
            U, S, Vt, tox_scores = _eigenshift_score_eigenvectors(
                model, toxic_acts, nontoxic_acts_collected, device)
            es_results = {}
            for frac in fracs:
                print(f"    frac={frac}…", end=" ", flush=True)
                _restore_ok = True
                try:
                    _apply_eigenshift(model, U, S, Vt, tox_scores, frac, device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    es_results[str(frac)] = _eval_and_record()
                    tox = es_results[str(frac)]["detoxify"]["toxicity"]["mean"]
                    ppl_r = es_results[str(frac)]["ppl_ratio"]
                    print(f"tox={tox:.4f}  ppl_ratio={ppl_r:.3f}")
                except RuntimeError as _exc:
                    if "CUDA" in str(_exc).upper() or "CUBLAS" in str(_exc).upper():
                        print(f"CUDA error at frac={frac}: {_exc!r}")
                        cuda_ok = _restore_ok = False
                    else:
                        raise
                finally:
                    if _restore_ok:
                        _restore_lm_head(model, lm_head_snap)
                if not cuda_ok:
                    break
            results["eigenshift"] = es_results
            _save_incremental()
            print(f"    [saved incrementally]")

    # ── 2. Self-Debiasing ─────────────────────────────────────────────────
    if not skip_self_debiasing:
        if "self_debiasing" in results and results["self_debiasing"]:
            print(f"  ── self_debiasing (already done, skipping) ──")
        elif not cuda_ok:
            print(f"  ── self_debiasing (skipped – GPU context corrupted) ──")
            results["self_debiasing"] = {}
        else:
            print(f"  ── self_debiasing ──")
            sd_results = {}
            for frac in fracs:
                alpha = frac * 2.0
                print(f"    frac={frac} (α={alpha:.2f})…", end=" ", flush=True)
                try:
                    res = evaluate_self_debiased(
                        model, tokenizer, prompts, detox_model, device,
                        owt_ref_text, alpha=alpha,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature, top_k=top_k,
                        llamaguard_scorer=llamaguard_scorer,
                    )
                    ppl_ratio = res["perplexity"] / max(base_ppl, 1e-6)
                    vl_ratio = res["val_loss"] / max(base_vl, 1e-6) \
                        if not math.isnan(res["val_loss"]) else float("nan")
                    sd_results[str(frac)] = {
                        **res, "ppl_ratio": ppl_ratio, "val_loss_ratio": vl_ratio,
                        "alpha": alpha}
                    tox = res["detoxify"]["toxicity"]["mean"]
                    print(f"tox={tox:.4f}  ppl_ratio={ppl_ratio:.3f}")
                except RuntimeError as _exc:
                    if "CUDA" in str(_exc).upper() or "CUBLAS" in str(_exc).upper():
                        print(f"CUDA error at frac={frac}: {_exc!r}")
                        cuda_ok = False
                    else:
                        raise
                if not cuda_ok:
                    break
            results["self_debiasing"] = sd_results
            _save_incremental()
            print(f"    [saved incrementally]")

    # ── 3. CHaRS-lite ─────────────────────────────────────────────────────
    if not skip_chars:
        if "chars_lite" in results and results["chars_lite"]:
            print(f"  ── chars_lite (already done, skipping) ──")
        elif not cuda_ok:
            print(f"  ── chars_lite (skipped – GPU context corrupted) ──")
            results["chars_lite"] = {}
        else:
            print(f"  ── chars_lite ──")
            chars_data = _compute_chars_directions(
                toxic_acts, nontoxic_acts_collected, n_clusters=5)
            ch_results = {}
            for frac in fracs:
                print(f"    frac={frac}…", end=" ", flush=True)
                _restore_ok = True
                try:
                    _apply_chars_steering(model, chars_data, frac, snap)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    ch_results[str(frac)] = _eval_and_record()
                    tox = ch_results[str(frac)]["detoxify"]["toxicity"]["mean"]
                    ppl_r = ch_results[str(frac)]["ppl_ratio"]
                    print(f"tox={tox:.4f}  ppl_ratio={ppl_r:.3f}")
                except RuntimeError as _exc:
                    if "CUDA" in str(_exc).upper() or "CUBLAS" in str(_exc).upper():
                        print(f"CUDA error at frac={frac}: {_exc!r}")
                        cuda_ok = _restore_ok = False
                    else:
                        raise
                finally:
                    if _restore_ok:
                        _restore_c_proj(model, snap)
                if not cuda_ok:
                    break
            results["chars_lite"] = ch_results
            _save_incremental()
            print(f"    [saved incrementally]")

    # ── 4. Vocab Shifting ─────────────────────────────────────────────────
    if not skip_vocab_shifting:
        if "vocab_shifting" in results and results["vocab_shifting"]:
            print(f"  ── vocab_shifting (already done, skipping) ──")
        elif not cuda_ok:
            print(f"  ── vocab_shifting (skipped – GPU context corrupted) ──")
            results["vocab_shifting"] = {}
        else:
            print(f"  ── vocab_shifting ──")
            toxic_token_scores = _compute_toxic_token_scores(
                model, prompts, nontoxic_texts, tokenizer, device)
            vs_results = {}
            for frac in fracs:
                alpha = frac * 5.0
                print(f"    frac={frac} (α={alpha:.2f})…", end=" ", flush=True)
                try:
                    res = evaluate_vocab_shifted(
                        model, tokenizer, prompts, toxic_token_scores,
                        detox_model, device, owt_ref_text,
                        alpha=alpha, max_new_tokens=max_new_tokens,
                        temperature=temperature, top_k=top_k,
                        llamaguard_scorer=llamaguard_scorer,
                    )
                    ppl_ratio = res["perplexity"] / max(base_ppl, 1e-6)
                    vl_ratio = res["val_loss"] / max(base_vl, 1e-6) \
                        if not math.isnan(res["val_loss"]) else float("nan")
                    vs_results[str(frac)] = {
                        **res, "ppl_ratio": ppl_ratio, "val_loss_ratio": vl_ratio,
                        "alpha": alpha}
                    tox = res["detoxify"]["toxicity"]["mean"]
                    print(f"tox={tox:.4f}  ppl_ratio={ppl_ratio:.3f}")
                except RuntimeError as _exc:
                    if "CUDA" in str(_exc).upper() or "CUBLAS" in str(_exc).upper():
                        print(f"CUDA error at frac={frac}: {_exc!r}")
                        cuda_ok = False
                    else:
                        raise
                if not cuda_ok:
                    break
            results["vocab_shifting"] = vs_results
            _save_incremental()
            print(f"    [saved incrementally]")

    # ── 5. PCT-OSD ────────────────────────────────────────────────────────
    if not skip_pct:
        if "pct_osd" in results and results["pct_osd"]:
            print(f"  ── pct_osd (already done, skipping) ──")
        elif not cuda_ok:
            print(f"  ── pct_osd (skipped – GPU context corrupted) ──")
            results["pct_osd"] = {}
        else:
            print(f"  ── pct_osd ──")
            pct_bases, pct_svals = _compute_pct_bases(
                toxic_acts, nontoxic_acts_collected)
            pct_results = {}
            for frac in fracs:
                print(f"    frac={frac}…", end=" ", flush=True)
                _restore_ok = True
                try:
                    _apply_pct_osd(model, pct_bases, pct_svals, frac, snap)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    pct_results[str(frac)] = _eval_and_record()
                    tox = pct_results[str(frac)]["detoxify"]["toxicity"]["mean"]
                    ppl_r = pct_results[str(frac)]["ppl_ratio"]
                    print(f"tox={tox:.4f}  ppl_ratio={ppl_r:.3f}")
                except RuntimeError as _exc:
                    if "CUDA" in str(_exc).upper() or "CUBLAS" in str(_exc).upper():
                        print(f"CUDA error at frac={frac}: {_exc!r}")
                        cuda_ok = _restore_ok = False
                    else:
                        raise
                finally:
                    if _restore_ok:
                        _restore_c_proj(model, snap)
                if not cuda_ok:
                    break
            results["pct_osd"] = pct_results
            _save_incremental()
            print(f"    [saved incrementally]")

    # ── Store heuristics ──────────────────────────────────────────────────
    heur = {}
    if toxic_acts and nontoxic_acts_collected:
        t_stats = compute_neuron_t_stats(toxic_acts, nontoxic_acts_collected)
        heur["t_stats"] = {str(li): ts.tolist() for li, ts in t_stats.items()}
    results["heuristics"] = heur
    _save_incremental()

    return results


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Evaluate additional detoxification strategies on "
                    "topo-nanoGPT 450M.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--taus", type=str,
                   default=",".join(str(t) for t in ALL_TAUS))
    p.add_argument("--step", type=int, default=FINAL_STEP)
    p.add_argument("--n_prompts", type=int, default=200)
    p.add_argument("--n_gen", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--fracs", type=str,
                   default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
    p.add_argument("--n_selectivity_tokens", type=int, default=4096)
    p.add_argument("--n_osd_components", type=int, default=32)
    p.add_argument("--n_clean_components", type=int, default=32)

    p.add_argument("--no_toxigen", action="store_true")
    p.add_argument("--no_rtp", action="store_true")

    p.add_argument("--no_eigenshift", action="store_true")
    p.add_argument("--no_self_debiasing", action="store_true")
    p.add_argument("--no_chars", action="store_true")
    p.add_argument("--no_vocab_shifting", action="store_true")
    p.add_argument("--no_pct", action="store_true")
    p.add_argument("--no_llamaguard", action="store_true")
    p.add_argument("--llamaguard_model", type=str, default=None)

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str,
                   default="outputs/toxicity_strategies_nanogpt_450m")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    taus = [int(t) for t in args.taus.split(",")]
    fracs = [float(f) for f in args.fracs.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = tiktoken.get_encoding("gpt2")

    rtp_prompts = tg_prompts = None
    if not args.no_rtp:
        rtp_prompts = load_realtoxicity_prompts(args.n_prompts)
        print(f"RTP prompts: {len(rtp_prompts)}")
    if not args.no_toxigen:
        tg_prompts = load_toxigen_prompts(args.n_prompts)
        print(f"ToxiGen prompts: {len(tg_prompts)}")
    nontoxic_texts = list(_NON_TOXIC_TEXTS) * 4

    owt_ref = _load_owt_reference()

    from detoxify import Detoxify
    detox_model = Detoxify("unbiased", device="cpu")

    llamaguard_scorer = None

    _requested = []
    if not args.no_eigenshift:
        _requested.append("eigenshift")
    if not args.no_self_debiasing:
        _requested.append("self_debiasing")
    if not args.no_chars:
        _requested.append("chars_lite")
    if not args.no_vocab_shifting:
        _requested.append("vocab_shifting")
    if not args.no_pct:
        _requested.append("pct_osd")

    all_results = {}

    for tau in taus:
        safe_tau = str(tau).replace(".", "_")
        label = f"tau{safe_tau}"
        out_json = output_dir / f"strategies_{label}.json"
        print(f"\n{'='*60}")
        print(f"  τ = {tau}  ({label})  step={args.step}")
        print(f"{'='*60}")

        # Resume
        tau_results = {}
        if args.resume and out_json.exists():
            with open(out_json) as f:
                tau_results = json.load(f)
            _all_done = True
            for ds_key in _DATASET_KEYS:
                ds = tau_results.get(ds_key, {})
                if not ds:
                    _all_done = False
                    break
                for mk in _requested:
                    if mk not in ds or not ds[mk]:
                        _all_done = False
                        break
                if not _all_done:
                    break
            if _all_done:
                print(f"  [resume] All requested methods done — skipping.")
                all_results[label] = tau_results
                continue
            else:
                print(f"  [resume] Partially complete — continuing.")

        # Load checkpoint (450M: config JSON + step checkpoint)
        run_name = f"gpt2-450m-tau-{tau}-downsample-9.0-all-topo"
        config_path = str(CKPT_ROOT / f"{run_name}.json")
        ckpt_path = str(CKPT_ROOT / run_name / f"step_{args.step:06d}.pt")
        model = load_gpt_checkpoint(config_path, ckpt_path, device)
        print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

        tau_results["tau"] = tau
        tau_results["label"] = label

        for ds_key, ds_prompts in [
            ("realtoxicityprompts", rtp_prompts),
            ("toxigen", tg_prompts),
        ]:
            if ds_prompts is None:
                continue
            print(f"\n  ── Dataset: {_DATASET_LABELS[ds_key]} "
                  f"({len(ds_prompts)} prompts) ──")

            existing_ds = tau_results.get(ds_key, {})
            if "baseline" in existing_ds and existing_ds["baseline"]:
                baseline = existing_ds["baseline"]
                print(f"  Baseline (from resume): "
                      f"tox={baseline['detoxify']['toxicity']['mean']:.4f}")
            else:
                print(f"  Baseline evaluation…")
                baseline = evaluate_model_full(
                    model=model, tokenizer=tokenizer, prompts=ds_prompts,
                    detox_model=detox_model, device=device,
                    owt_ref_text=owt_ref,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_k=args.top_k,
                    n_gen=args.n_gen, llamaguard_scorer=llamaguard_scorer,
                )
                print(f"  Baseline tox={baseline['detoxify']['toxicity']['mean']:.4f} "
                      f"ppl={baseline['perplexity']:.2f}")

            # Incremental save closure — called after each method completes
            def _save_incremental(sweep_dict, _ds=ds_key):
                tau_results[_ds] = sweep_dict
                with open(out_json, "w") as _f:
                    json.dump(tau_results, _f, indent=2, allow_nan=True)

            sweep_result = run_strategy_sweep(
                model=model, tokenizer=tokenizer, prompts=ds_prompts,
                nontoxic_texts=nontoxic_texts, detox_model=detox_model,
                device=device, baseline_result=baseline,
                owt_ref_text=owt_ref, fracs=fracs,
                n_selectivity_tokens=args.n_selectivity_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k,
                n_gen=args.n_gen, llamaguard_scorer=llamaguard_scorer,
                skip_eigenshift=args.no_eigenshift,
                skip_self_debiasing=args.no_self_debiasing,
                skip_chars=args.no_chars,
                skip_vocab_shifting=args.no_vocab_shifting,
                skip_pct=args.no_pct,
                existing=existing_ds,
                save_fn=_save_incremental,
            )
            tau_results[ds_key] = sweep_result

        # Final save per-tau
        with open(out_json, "w") as f:
            json.dump(tau_results, f, indent=2, allow_nan=True)
        print(f"  Saved → {out_json}")
        all_results[label] = tau_results

        del model
        torch.cuda.empty_cache()

    print(f"\nAll done. Results in: {output_dir}/")


if __name__ == "__main__":
    main()
