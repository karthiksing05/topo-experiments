"""
eval_toxicity_nanogpt_quantized.py
───────────────────────────────────
Post-training-quantized version of the nanoGPT toxicity benchmark.

Each checkpoint is loaded from murtylab/topo-nanogpt-fineweb10B, then
quantized before evaluation.  All experiments are identical to
eval_toxicity_nanogpt.py; this script exists to compare model behaviour
under reduced precision.

Outputs are saved to outputs/toxicity_nanogpt_quantized/.

Supported quantization modes (--quantization):
  fp16  (default) — model.half()
                    GPU Tensor-Core accelerated; 2× memory saving; minimal
                    accuracy loss for inference.
  bf16            — model.to(bfloat16)
                    Same size as fp16 but wider dynamic range; recommended
                    on Ampere (A100) or newer GPUs.
  int8            — bitsandbytes 8-bit matrix-multiply (LLM.int8()).
                    Requires the `bitsandbytes` package; falls back to fp16
                    if unavailable.  Provides ~4× memory saving with a small
                    accuracy trade-off.

All three modes run the forward pass natively on CUDA, utilising Tensor
Cores (fp16/bf16) or dedicated int8 GEMM kernels (int8).

Usage:
  python src/test/eval_toxicity_nanogpt_quantized.py \
      [--quantization fp16|bf16|int8] \
      [--n_prompts 200] [--n_gen 1] [--max_new_tokens 200] \
      [--taus 0.0,0.5,1.0,3.0,50.0] \
      [--no_pruning] [--no_svd_pruning] [--no_amplification]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

# ── Import shared logic from the base eval script ─────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from eval_toxicity_nanogpt import (  # noqa: E402
    GPT,
    load_gpt_checkpoint,
    load_toxic_prompts,
    evaluate_model,
    run_toxicity_pruning,
    run_toxicity_amplification,
    run_svd_pruning,
    compute_svd_selectivity,
    compute_neuron_selectivity,
    collect_mlp_activations,
    plot_comparison,
    plot_pruning_comparison,
    plot_amplification_comparison,
    plot_effective_rank,
    plot_svd_pruning_comparison,
    plot_svd_cross_model_comparison,
    save_selectivity_visualizations,
    save_svd_visualizations,
    save_amplification_visualizations,
    _NON_TOXIC_TEXTS,
    HF_REPO,
    BASELINE_TAU,
    HF_CACHE,
    effective_rank,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]

# ── Quantization helpers ───────────────────────────────────────────────────────

SUPPORTED_QUANT = ("fp16", "bf16", "int8", "int4")


def apply_quantization(model: GPT, quant_type: str, device: torch.device) -> GPT:
    """Apply post-training quantization to *model* and return it.

    fp16
        Cast all parameters to float16 with ``model.half()``.
        Matrix multiplications execute via CUDA Tensor Cores for accelerated
        throughput on any GPU with compute capability ≥ 5.3.

    bf16
        Cast all parameters to bfloat16 (same 16-bit width as fp16 but with
        the exponent range of float32 to avoid under/overflow).  Best on
        Ampere (A100) or newer GPUs which have native bf16 Tensor Cores.
        Falls back to fp16 on older devices.

    int8
        Replace every ``nn.Linear`` with a ``bitsandbytes.nn.Linear8bitLt``
        that performs 8-bit GEMM on CUDA (LLM.int8() algorithm from
        Dettmers et al. 2022).  Requires the ``bitsandbytes`` package.
        Falls back to fp16 if the package is not installed.

    int4
        Replace every ``nn.Linear`` with a ``bitsandbytes.nn.Linear4bit``
        using NF4 (normalized float-4) quantization with double-quantization
        enabled (QLoRA, Dettmers et al. 2023).  Provides ~8× memory saving
        vs float32.  Requires ``bitsandbytes`` ≥ 0.39.0 and CUDA CC ≥ 7.5.
        Falls back to fp16 if the package is not installed.
    """
    if quant_type == "fp16":
        return model.half()

    if quant_type == "bf16":
        if device.type != "cuda":
            print("  WARNING: bf16 requires CUDA; falling back to fp16.")
            return model.half()
        return model.to(torch.bfloat16)

    if quant_type == "int8":
        if device.type != "cuda":
            print("  WARNING: int8 (bitsandbytes) requires CUDA; falling back to fp16.")
            return model.half()
        try:
            import bitsandbytes as bnb  # noqa: PLC0415

            def _replace_linear(module: nn.Module) -> None:
                for name, child in list(module.named_children()):
                    if isinstance(child, nn.Linear):
                        new_layer = bnb.nn.Linear8bitLt(
                            child.in_features,
                            child.out_features,
                            bias=child.bias is not None,
                            has_fp16_weights=False,
                            threshold=6.0,   # outlier threshold from LLM.int8()
                        )
                        # Copy fp16 data INTO the existing Int8Params object —
                        # do NOT replace .weight with a plain nn.Parameter, as
                        # Int8Params carries the .CB / .SCB attributes that
                        # Linear8bitLt.forward() requires.
                        new_layer.weight.data = child.weight.half().data
                        if child.bias is not None:
                            new_layer.bias = nn.Parameter(child.bias.half().data)
                        setattr(module, name, new_layer)
                    else:
                        _replace_linear(child)

            _replace_linear(model)
            # Move to device — this triggers Int8Params.__cuda__ which runs
            # the CB quantization pass and populates .CB / .SCB.
            model.to(device)
            print(f"  int8: replaced nn.Linear layers with bnb.Linear8bitLt "
                  f"(bitsandbytes {bnb.__version__})")
            return model

        except ImportError:
            print("  WARNING: bitsandbytes not installed; falling back to fp16.")
            return model.half()

    if quant_type == "int4":
        if device.type != "cuda":
            print("  WARNING: int4 (bitsandbytes) requires CUDA; falling back to fp16.")
            return model.half()
        try:
            import bitsandbytes as bnb  # noqa: PLC0415

            def _replace_linear_4bit(module: nn.Module) -> None:
                for name, child in list(module.named_children()):
                    if isinstance(child, nn.Linear):
                        new_layer = bnb.nn.Linear4bit(
                            child.in_features,
                            child.out_features,
                            bias=child.bias is not None,
                            quant_type="nf4",          # NF4: optimal for normal-distributed weights
                            compress_statistics=True,  # double-quantization (QLoRA)
                        )
                        # Linear4bit stores weights as fp16 initially; quantization
                        # happens on the first forward pass on CUDA.
                        new_layer.weight = bnb.nn.Params4bit(
                            child.weight.half().data,
                            requires_grad=False,
                            quant_type="nf4",
                        )
                        if child.bias is not None:
                            new_layer.bias = nn.Parameter(child.bias.half().data)
                        setattr(module, name, new_layer)
                    else:
                        _replace_linear_4bit(child)

            _replace_linear_4bit(model)
            # Moving to device triggers bitsandbytes to finalise the NF4
            # quantization state (quant_state) for every Linear4bit layer.
            # Without this the first forward pass hits an AssertionError inside
            # fix_4bit_weight_quant_state_from_module.
            model.to(device)
            print(f"  int4: replaced nn.Linear layers with bnb.Linear4bit (NF4+DQ) "
                  f"(bitsandbytes {bnb.__version__})")
            return model

        except ImportError:
            print("  WARNING: bitsandbytes not installed; falling back to fp16.")
            return model.half()

    raise ValueError(
        f"Unknown quantization type {quant_type!r}. Choose from {SUPPORTED_QUANT}."
    )


# ── Hook-based pruning/amplification for packed-weight quantizations ──────────
# int8/int4 store weights in an opaque packed format; in-place indexed
# assignment (weight[:, idx] = 0.0) is undefined on those layers.
# The functions below achieve the same effect by registering forward hooks
# on each MLP's GELU module that zero or scale the post-GELU activations.
# Zeroing position j in the post-GELU hidden state is mathematically
# equivalent to zeroing column j of c_proj.weight.

def prune_and_restore_quantized(
    model:    GPT,
    t_stats:  dict,
    fraction: float,
    fn,
):
    """Hook-based equivalent of prune_and_restore for quantized models."""
    hooks = []
    for layer_idx, block in enumerate(model.transformer.h):
        t = t_stats.get(layer_idx)
        if t is None:
            continue
        n_prune  = max(1, int(len(t) * fraction))
        top_idx  = np.argsort(t)[-n_prune:]
        mask     = torch.ones(len(t), dtype=torch.float32)
        mask[top_idx] = 0.0

        def _make_hook(m):
            def _hook(module, inp, out):
                return out * m.to(dtype=out.dtype, device=out.device)
            return _hook

        hooks.append(block.mlp.gelu.register_forward_hook(_make_hook(mask)))

    try:
        return fn(model)
    finally:
        for h in hooks:
            h.remove()


def amplify_and_restore_quantized(
    model:    GPT,
    t_stats:  dict,
    fraction: float,
    factor:   float,
    fn,
):
    """Hook-based equivalent of amplify_and_restore for quantized models."""
    hooks = []
    for layer_idx, block in enumerate(model.transformer.h):
        t = t_stats.get(layer_idx)
        if t is None:
            continue
        n_amp   = max(1, int(len(t) * fraction))
        top_idx = np.argsort(t)[-n_amp:]
        scale   = torch.ones(len(t), dtype=torch.float32)
        scale[top_idx] = float(factor)

        def _make_hook(s):
            def _hook(module, inp, out):
                return out * s.to(dtype=out.dtype, device=out.device)
            return _hook

        hooks.append(block.mlp.gelu.register_forward_hook(_make_hook(scale)))

    try:
        return fn(model)
    finally:
        for h in hooks:
            h.remove()


def run_toxicity_pruning_quantized(
    model,
    tokenizer,
    toxic_prompts,
    nontoxic_texts,
    detox_model,
    device,
    baseline_result,
    pruning_fracs,
    max_new_tokens   = 200,
    temperature      = 1.0,
    top_k            = 50,
    n_gen            = 1,
    n_selectivity_tokens = 4096,
) -> dict:
    """Mirror of run_toxicity_pruning using activation-hook pruning for
    quantized models (int8/int4) where weight assignment is not possible."""
    max_toks_per_text = max(1, n_selectivity_tokens // max(1, len(toxic_prompts)))
    print(f"    Collecting activations ({len(toxic_prompts)} toxic, "
          f"{len(nontoxic_texts)} non-toxic, {max_toks_per_text} tok/text)…")
    toxic_acts    = collect_mlp_activations(model, toxic_prompts,  tokenizer, device, max_toks_per_text)
    nontoxic_acts = collect_mlp_activations(model, nontoxic_texts, tokenizer, device, max_toks_per_text)
    t_stats, global_stats = compute_neuron_selectivity(toxic_acts, nontoxic_acts)
    print(f"    Selectivity: {global_stats['frac_significant_t2']*100:.1f}% of neurons have t>2")

    baseline_ppl   = baseline_result["perplexity"]
    pruned_results: dict = {}

    for frac in pruning_fracs:
        pct = frac * 100
        print(f"    Pruning {pct:.0f}% (act. mask)…", end=" ", flush=True)

        def _eval(m, _f=frac):
            return evaluate_model(
                model=m, tokenizer=tokenizer, prompts=toxic_prompts,
                detox_model=detox_model, device=device,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_k=top_k, n_gen=n_gen,
            )

        res       = prune_and_restore_quantized(model, t_stats, frac, _eval)
        ppl_ratio = res["perplexity"] / max(baseline_ppl, 1e-6)
        tox_mean  = res["toxicity_scores"]["toxicity"]["mean"]
        print(f"tox={tox_mean:.4f}  ppl_ratio={ppl_ratio:.3f}")
        pruned_results[str(frac)] = {
            "toxicity_scores": res["toxicity_scores"],
            "perplexity":      res["perplexity"],
            "ppl_ratio":       ppl_ratio,
        }

    return {
        "pruning_fractions":  pruning_fracs,
        "unpruned": {
            "toxicity_scores": baseline_result["toxicity_scores"],
            "ppl":             baseline_ppl,
        },
        "pruned":         pruned_results,
        "neuron_stats":   global_stats,
        "t_stats_per_layer": {str(k): v.tolist() for k, v in t_stats.items()},
    }


def run_toxicity_amplification_quantized(
    model,
    tokenizer,
    toxic_prompts,
    nontoxic_texts,
    detox_model,
    device,
    baseline_result,
    amp_fracs,
    amp_factor           = 5.0,
    t_stats              = None,
    max_new_tokens       = 200,
    temperature          = 1.0,
    top_k                = 50,
    n_gen                = 1,
    n_selectivity_tokens = 4096,
) -> dict:
    """Mirror of run_toxicity_amplification using activation-hook scaling for
    quantized models (int8/int4) where weight assignment is not possible."""
    if t_stats is None:
        max_toks_per_text = max(1, n_selectivity_tokens // max(1, len(toxic_prompts)))
        print(f"    Collecting activations for amplification ({max_toks_per_text} tok/text)…")
        toxic_acts    = collect_mlp_activations(model, toxic_prompts,  tokenizer, device, max_toks_per_text)
        nontoxic_acts = collect_mlp_activations(model, nontoxic_texts, tokenizer, device, max_toks_per_text)
        t_stats, _ = compute_neuron_selectivity(toxic_acts, nontoxic_acts)

    baseline_ppl = baseline_result["perplexity"]
    amp_results: dict = {}

    for frac in amp_fracs:
        pct = frac * 100
        print(f"    Amplifying {pct:.0f}% ×{amp_factor}× (act. scale)…", end=" ", flush=True)

        def _eval(m):
            return evaluate_model(
                model=m, tokenizer=tokenizer, prompts=toxic_prompts,
                detox_model=detox_model, device=device,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_k=top_k, n_gen=n_gen,
            )

        res       = amplify_and_restore_quantized(model, t_stats, frac, amp_factor, _eval)
        tox_mean  = res["toxicity_scores"]["toxicity"]["mean"]
        ppl_ratio = res["perplexity"] / max(baseline_ppl, 1e-6)
        print(f"tox={tox_mean:.4f}  ppl_ratio={ppl_ratio:.3f}")
        amp_results[str(frac)] = {
            "toxicity_scores": res["toxicity_scores"],
            "perplexity":      res["perplexity"],
            "ppl_ratio":       ppl_ratio,
        }

    return {
        "amp_fracs":  amp_fracs,
        "amp_factor": amp_factor,
        "unpruned": {
            "toxicity_scores": baseline_result["toxicity_scores"],
            "ppl":             baseline_ppl,
        },
        "amplified": amp_results,
    }


def dtype_str(quant_type: str) -> str:
    """Human-readable dtype label for plot annotations."""
    return {"fp16": "float16", "bf16": "bfloat16", "int8": "int8", "int4": "nf4 (4-bit)"}[quant_type]


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantized toxicity benchmark for topo-nanoGPT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_prompts",      type=int,   default=200)
    parser.add_argument("--n_gen",          type=int,   default=1)
    parser.add_argument("--max_new_tokens", type=int,   default=200)
    parser.add_argument("--temperature",    type=float, default=1.0)
    parser.add_argument("--top_k",          type=int,   default=50,
                        help="0 = greedy")
    parser.add_argument("--taus",           type=str,   default="0.0,0.5,1.0,3.0,50.0")
    parser.add_argument("--device",         type=str,   default=None)
    parser.add_argument("--pruning_fracs",  type=str,   default="0.05,0.1,0.15,0.2")
    parser.add_argument("--n_selectivity_tokens", type=int, default=4096)
    parser.add_argument("--no_pruning",     action="store_true")
    parser.add_argument("--svd_pruning_fracs", type=str, default="0.05,0.1,0.15,0.2")
    parser.add_argument("--no_svd_pruning",  action="store_true")
    parser.add_argument("--amp_factor",     type=float, default=5.0)
    parser.add_argument("--amp_fracs",      type=str,   default="0.05,0.1,0.15,0.2")
    parser.add_argument("--no_amplification", action="store_true")
    parser.add_argument(
        "--quantization",
        type=str,
        default="fp16",
        choices=list(SUPPORTED_QUANT),
        help="Post-training quantization mode applied to every checkpoint. "
             "int4 uses NF4 + double-quantization (QLoRA); requires bitsandbytes.",
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    taus   = [float(t) for t in args.taus.split(",")]
    top_k  = args.top_k if args.top_k > 0 else None
    pruning_fracs     = [float(f) for f in args.pruning_fracs.split(",")]
    svd_pruning_fracs = [float(f) for f in args.svd_pruning_fracs.split(",")]
    amp_fracs         = [float(f) for f in args.amp_fracs.split(",")]
    amp_factor        = args.amp_factor
    quant_type        = args.quantization

    # int8/int4 use bitsandbytes packed weights that don't support in-place
    # indexed assignment.  Neuron pruning and amplification are transparently
    # re-routed through forward hooks (activation masking/scaling) which work
    # with any weight format.  SVD pruning modifies weight matrices in a
    # structured low-rank way with no hook equivalent, so it remains disabled
    # for packed quantizations.
    USE_HOOK_PRUNING = quant_type in ("int8", "int4")
    if USE_HOOK_PRUNING and not args.no_svd_pruning:
        print(
            f"  NOTE: SVD pruning disabled for {quant_type} (requires mutable float "
            f"weights). Neuron pruning and amplification use activation hooks instead."
        )
        args.no_svd_pruning = True

    # Each quantization mode writes to its own subdirectory so runs never
    # overwrite each other.
    OUTPUT_DIR = BASE_DIR / "outputs" / f"toxicity_nanogpt_quantized_{quant_type}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device          : {device}")
    print(f"Quantization    : {quant_type}  ({dtype_str(quant_type)})")
    print(f"Pruning method  : {'activation hooks (packed-weight compat.)' if USE_HOOK_PRUNING else 'weight column zeroing'}")
    print(f"Taus            : {taus}")
    print(f"Prompts         : {args.n_prompts}  ×  {args.n_gen} generation(s) each")
    print(f"Pruning fracs   : {pruning_fracs if not args.no_pruning else 'DISABLED'}")
    print(f"SVD prune fracs : {svd_pruning_fracs if not args.no_svd_pruning else 'DISABLED'}")
    print(f"Amp factor/fracs: {amp_factor}× @ {amp_fracs if not args.no_amplification else 'DISABLED'}")
    print(f"Selectivity toks: {args.n_selectivity_tokens}")
    print(f"Output dir      : {OUTPUT_DIR}")
    print()

    # ── Load tokenizer ─────────────────────────────────────────────────────
    import tiktoken  # noqa: PLC0415
    tokenizer = tiktoken.get_encoding("gpt2")

    # ── Load detoxify once on CPU to save GPU memory ───────────────────────
    print("Loading Detoxify (unbiased model)…")
    from detoxify import Detoxify  # noqa: PLC0415
    detox_model = Detoxify("unbiased", device="cpu")
    print()

    # ── Load prompts ───────────────────────────────────────────────────────
    prompts  = load_toxic_prompts(args.n_prompts)
    nontoxic = list(_NON_TOXIC_TEXTS)

    SEL_DIR = OUTPUT_DIR / "selectivity"

    results             = {}
    pruning_results     = {}
    amp_results         = {}
    svd_pruning_results = {}
    svd_selectivity_all = {}

    for tau in taus:
        label = f"tau={tau} [{quant_type}]" if tau != BASELINE_TAU \
                else f"tau={tau} (baseline) [{quant_type}]"
        print(f"=== {label} ===")

        # Download checkpoint
        filename = f"tau_{tau}.pt"
        print(f"  Downloading {filename} from {HF_REPO}…")
        ckpt_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            cache_dir=str(HF_CACHE),
        )

        # Load and quantize
        print(f"  Loading model…")
        model = load_gpt_checkpoint(ckpt_path, device)
        print(f"  Applying {quant_type} quantization…")
        model = apply_quantization(model, quant_type, device)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        param_dtype = next(model.parameters()).dtype
        print(f"  Model loaded ({n_params:.1f}M params, dtype={param_dtype})")

        print(f"  Generating continuations…")
        result = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            detox_model=detox_model,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            n_gen=args.n_gen,
        )
        tox_mean = result["toxicity_scores"]["toxicity"]["mean"]
        tox_p95  = result["toxicity_scores"]["toxicity"]["p95"]
        print(f"  Toxicity — mean={tox_mean:.4f}  p95={tox_p95:.4f}  PPL={result['perplexity']:.2f}")
        results[label] = result

        # ── Toxicity pruning sweep ─────────────────────────────────────────
        prun_result = None
        if not args.no_pruning:
            _run_pruning = run_toxicity_pruning_quantized if USE_HOOK_PRUNING else run_toxicity_pruning
            print(f"  Running toxicity pruning sweep ({len(pruning_fracs)} fracs"
                  f"{', hook-based' if USE_HOOK_PRUNING else ''})…")
            prun_result = _run_pruning(
                model=model,
                tokenizer=tokenizer,
                toxic_prompts=prompts,
                nontoxic_texts=nontoxic,
                detox_model=detox_model,
                device=device,
                baseline_result=result,
                pruning_fracs=pruning_fracs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
                n_gen=args.n_gen,
                n_selectivity_tokens=args.n_selectivity_tokens,
            )
            pruning_results[label] = prun_result

            safe_tau = str(tau).replace(".", "_")
            prun_path = OUTPUT_DIR / f"pruning_tau{safe_tau}.json"
            prun_json = {k: v for k, v in prun_result.items() if k != "t_stats_per_layer"}
            with open(prun_path, "w") as f:
                json.dump(prun_json, f, indent=2)
            print(f"  Pruning results → {prun_path}")

            t_stats_path = OUTPUT_DIR / f"t_stats_tau{safe_tau}.json"
            with open(t_stats_path, "w") as f:
                json.dump(prun_result["t_stats_per_layer"], f)
            print(f"  t-statistics    → {t_stats_path}")

            # Selectivity visualizations
            print(f"  Saving selectivity visualizations…")
            safe_label = label.replace(" ", "_").replace("=", "")
            vis_dir = SEL_DIR / safe_label
            t_stats_np = {
                int(k): np.array(v)
                for k, v in prun_result["t_stats_per_layer"].items()
            }
            save_selectivity_visualizations(
                t_stats_per_layer=t_stats_np,
                global_stats=prun_result["neuron_stats"],
                pruning_result=prun_result,
                label=label,
                vis_dir=vis_dir,
            )

            # SVD selectivity & pruning
            if not args.no_svd_pruning:
                print(f"  Computing SVD selectivity…")
                svd_sel = compute_svd_selectivity(model, t_stats_np)
                svd_selectivity_all[label] = svd_sel

                svd_sel_path = OUTPUT_DIR / f"svd_selectivity_tau{safe_tau}.json"
                svd_sel_serialisable = {
                    str(li): {
                        "singular_values":  info["singular_values"],
                        "component_scores": info["component_scores"],
                        "effective_rank":   info["effective_rank"],
                        "cum_variance":     info["cum_variance"],
                    }
                    for li, info in svd_sel.items()
                }
                with open(svd_sel_path, "w") as f:
                    json.dump(svd_sel_serialisable, f)
                print(f"  SVD selectivity → {svd_sel_path}")

                print(f"  Running SVD pruning sweep ({len(svd_pruning_fracs)} fracs)…")
                svd_prun = run_svd_pruning(
                    model=model,
                    tokenizer=tokenizer,
                    toxic_prompts=prompts,
                    nontoxic_texts=nontoxic,
                    detox_model=detox_model,
                    device=device,
                    baseline_result=result,
                    t_stats=t_stats_np,
                    svd_selectivity=svd_sel,
                    pruning_fracs=svd_pruning_fracs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=top_k,
                    n_gen=args.n_gen,
                )
                svd_pruning_results[label] = svd_prun

                svd_prun_path = OUTPUT_DIR / f"svd_pruning_tau{safe_tau}.json"
                with open(svd_prun_path, "w") as f:
                    json.dump(svd_prun, f, indent=2)
                print(f"  SVD pruning     → {svd_prun_path}")

                print(f"  Saving SVD visualizations…")
                save_svd_visualizations(
                    svd_sel=svd_sel_serialisable,
                    svd_prun=svd_prun,
                    label=label,
                    vis_dir=SEL_DIR / safe_label,
                )

        # ── Toxic-neuron amplification sweep ────────────────────────────────
        if not args.no_amplification and prun_result is not None:
            _run_amp = run_toxicity_amplification_quantized if USE_HOOK_PRUNING else run_toxicity_amplification
            print(f"  Running amplification sweep ({len(amp_fracs)} fracs, ×{amp_factor}"
                  f"{', hook-based' if USE_HOOK_PRUNING else ''})…")
            t_stats_np = {
                int(k): np.array(v)
                for k, v in prun_result["t_stats_per_layer"].items()
            }
            amp_result = _run_amp(
                model=model,
                tokenizer=tokenizer,
                toxic_prompts=prompts,
                nontoxic_texts=nontoxic,
                detox_model=detox_model,
                device=device,
                baseline_result=result,
                amp_fracs=amp_fracs,
                amp_factor=amp_factor,
                t_stats=t_stats_np,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
                n_gen=args.n_gen,
            )
            amp_results[label] = amp_result

            safe_tau = str(tau).replace(".", "_")
            amp_path = OUTPUT_DIR / f"amplification_tau{safe_tau}.json"
            with open(amp_path, "w") as f:
                json.dump(amp_result, f, indent=2)
            print(f"  Amplification   → {amp_path}")

            print(f"  Saving amplification visualizations…")
            save_amplification_visualizations(
                amp_result=amp_result,
                pruning_result=prun_result,
                t_stats_per_layer=t_stats_np,
                label=label,
                vis_dir=SEL_DIR / label.replace(" ", "_").replace("=", ""),
            )

        # Free GPU memory before loading the next model
        del model
        torch.cuda.empty_cache()
        print()

    # ── Save summary JSON ──────────────────────────────────────────────────
    json_path = OUTPUT_DIR / "results.json"
    summary = {
        key: {
            "quantization":    quant_type,
            "n_prompts":       r["n_prompts"],
            "n_completions":   r["n_completions"],
            "toxicity_scores": r["toxicity_scores"],
            "perplexity":      r["perplexity"],
        }
        for key, r in results.items()
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved → {json_path}")

    # ── Print summary table ────────────────────────────────────────────────
    quant_header = f"[{quant_type}]"
    print(f"\n── Summary ({quant_header}) ──────────────────────────────────────────")
    header = f"{'Model':<38}  {'Mean Tox':>10}  {'p95 Tox':>10}  {'Max Tox':>10}  {'PPL':>8}"
    print(header)
    print("─" * len(header))
    for key, r in results.items():
        ts  = r["toxicity_scores"]["toxicity"]
        ppl = r["perplexity"]
        print(f"{key:<38}  {ts['mean']:>10.4f}  {ts['p95']:>10.4f}  {ts['max']:>10.4f}  {ppl:>8.2f}")
    print()

    # ── Cross-model plots ──────────────────────────────────────────────────
    if svd_selectivity_all:
        print("Plotting effective rank and SVD spectra…")
        plot_effective_rank(svd_selectivity_all, OUTPUT_DIR)
    if svd_pruning_results and pruning_results:
        print("Plotting SVD vs. neuron pruning comparison…")
        plot_svd_pruning_comparison(svd_pruning_results, pruning_results, OUTPUT_DIR)
    if svd_pruning_results:
        print("Plotting SVD cross-model comparison…")
        plot_svd_cross_model_comparison(svd_pruning_results, OUTPUT_DIR)

    if pruning_results:
        print("── Pruning summary (mean tox after 20% pruning) ─────────────────")
        for key, tp in pruning_results.items():
            f20 = tp["pruned"].get("0.2", tp["pruned"].get("0.20"))
            if f20:
                tox_20 = f20["toxicity_scores"]["toxicity"]["mean"]
                ppl_20 = f20["ppl_ratio"]
                base_t = tp["unpruned"]["toxicity_scores"]["toxicity"]["mean"]
                print(
                    f"  {key:<36}  tox@20%={tox_20:.4f} "
                    f"({tox_20 / max(base_t, 1e-8) * 100:.1f}%)  "
                    f"ppl_ratio={ppl_20:.3f}"
                )
        print()

    print("Saving comparison plots…")
    plot_comparison(results, OUTPUT_DIR)
    if pruning_results:
        print("Plotting pruning comparison…")
        plot_pruning_comparison(pruning_results, OUTPUT_DIR)
    if amp_results:
        print("Plotting amplification comparison…")
        plot_amplification_comparison(amp_results, pruning_results, OUTPUT_DIR)

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
