"""train_gpt2_topo_quant.py
===========================
Train a GPT-2-small (124M) from scratch on FineWeb-edu with TopoQuantLoss
applied to the MLP feedforward layers (c_fc / c_proj in every transformer block).

This matches the architecture of openai-community/gpt2, which has a widely
used 4-bit GPTQ counterpart (TheBloke/gpt2-GPTQ).  Training from scratch with
TopoQuantLoss should produce checkpoints that are naturally more amenable to
low-bit quantisation than a standard baseline.

Architecture:  GPT-2 small (12 layers, 12 heads, 768 embd, 1024 ctx)
Corpus:        HuggingFace HF dataset  "HuggingFaceFW/fineweb-edu"  (sample-10BT)
Tokeniser:     GPT-2 BPE (tiktoken cl100k_base / gpt2)

Two training recipes are run:
  baseline        — cross-entropy language modelling only
  topo_quant      — cross-entropy + SoftTopoQuantLoss on MLP layers

Outputs
-------
  outputs/topo_quant_gpt2/
    baseline/        checkpoint_latest.pt, log.json
    topo_quant/      checkpoint_latest.pt, log.json
  outputs/topo_quant_gpt2/training_curves.png

Usage
-----
    python src/topo_quant/train_gpt2_topo_quant.py \\
        --config configs/topo_quant_gpt2.json [--variant baseline|topo_quant] \\
        [--device cuda:0] [--max-steps 100000]
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

BASE_DIR   = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs" / "topo_quant_gpt2"
HF_CACHE   = BASE_DIR / ".hf_cache"
sys.path.insert(0, str(BASE_DIR / "src" / "topo_quant"))

from topo_quant_loss import SoftTopoQuantLoss

# ---------------------------------------------------------------------------
# GPT-2 architecture (matches openai-community/gpt2 exactly — no bias)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head  = n_head
        self.n_embd  = n_embd
        self.c_attn  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj  = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y   = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.c_fc   = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=False)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd, bias=False)
        self.mlp  = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """GPT-2 small — architecture matches openai-community/gpt2 (no bias)."""

    def __init__(
        self,
        vocab_size:  int   = 50304,   # padded to nice multiple
        n_layer:     int   = 12,
        n_head:      int   = 12,
        n_embd:      int   = 768,
        block_size:  int   = 1024,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(vocab_size, n_embd),
            wpe  = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            h    = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd, bias=False),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialise weights
        self.apply(self._init_weights)
        # Scale residual projections per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        pos  = torch.arange(T, device=idx.device)
        x    = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    def count_params(self) -> int:
        # Exclude weight-tied lm_head
        return sum(p.numel() for n, p in self.named_parameters() if "lm_head" not in n)


# ---------------------------------------------------------------------------
# Data — FineWeb-edu streamed via HuggingFace datasets
# ---------------------------------------------------------------------------

def build_data_iter(split: str, block_size: int, batch_size: int, hf_cache: Path):
    """Return an iterator that yields (x, y) token batches from fineweb-edu."""
    import datasets as hf_datasets

    os.environ.setdefault("HF_HOME",              str(hf_cache))
    os.environ.setdefault("HF_DATASETS_CACHE",    str(hf_cache / "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache / "hub"))

    enc = tiktoken.get_encoding("gpt2")

    # Stream so we never materialise the full corpus on disk
    ds = hf_datasets.load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    buffer: list[int] = []
    for sample in ds:
        tokens = enc.encode_ordinary(sample["text"])
        buffer.extend(tokens)
        buffer.append(enc.eot_token)
        # Yield complete chunks
        while len(buffer) >= (block_size + 1) * batch_size:
            chunk = buffer[: (block_size + 1) * batch_size]
            buffer = buffer[(block_size + 1) * batch_size :]
            t = torch.tensor(chunk, dtype=torch.long).view(batch_size, block_size + 1)
            yield t[:, :block_size], t[:, 1:]


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    decay = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return min_lr + coeff * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, optimizer, step: int, log: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step":       step,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }
    torch.save(ckpt, out_dir / "checkpoint_latest.pt")
    with open(out_dir / "log.json", "w") as f:
        json.dump(log, f, indent=2)


def load_checkpoint(model: nn.Module, optimizer, out_dir: Path) -> int:
    ckpt_path = out_dir / "checkpoint_latest.pt"
    if not ckpt_path.exists():
        return 0
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, variant: str) -> None:
    device   = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    out_dir  = OUTPUT_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Training GPT-2 small  |  variant: {variant}")
    print(f"  Device: {device}")
    print(f"  Output: {out_dir}")
    print(f"{'='*65}\n")

    # Model
    model = GPT2(
        vocab_size  = cfg.get("vocab_size", 50304),
        n_layer     = cfg.get("n_layer",    12),
        n_head      = cfg.get("n_head",     12),
        n_embd      = cfg.get("n_embd",     768),
        block_size  = cfg.get("block_size", 1024),
        dropout     = cfg.get("dropout",    0.0),
    ).to(device)
    print(f"  Parameters: {model.count_params():,}")

    # Compile (torch >= 2.0)
    if cfg.get("compile", True) and hasattr(torch, "compile"):
        print("  Compiling model with torch.compile ...")
        model = torch.compile(model)

    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["max_lr"],
        betas        = (cfg.get("beta1", 0.9), cfg.get("beta2", 0.95)),
        weight_decay = cfg.get("weight_decay", 0.1),
    )

    # Resume if available
    start_step = load_checkpoint(model, optimizer, out_dir)
    if start_step > 0:
        print(f"  Resumed from step {start_step}")

    max_steps     = cfg["max_steps"]
    block_size    = cfg.get("block_size", 1024)
    batch_size    = cfg.get("batch_size", 8)
    warmup_steps  = cfg.get("warmup_steps", 2000)
    log_interval  = cfg.get("log_interval", 100)
    save_interval = cfg.get("save_interval", 1000)
    grad_clip     = cfg.get("grad_clip",    1.0)
    num_bits      = cfg.get("num_bits", 4)

    # TopoQuantLoss only for topo_quant variant
    quant_fn = None
    if variant == "topo_quant":
        quant_fn = SoftTopoQuantLoss(
            num_bits            = num_bits,
            tau                 = cfg.get("tau", 1.0),
            initial_temperature = cfg.get("soft_initial_temp", 10.0),
            final_temperature   = cfg.get("soft_final_temp",   0.1),
            anneal_steps        = int(max_steps * cfg.get("anneal_fraction", 0.5)),
            anneal_schedule     = cfg.get("anneal_schedule", "cosine"),
            apply_to_layers     = ["c_fc", "c_proj"],   # MLP layers only
        ).to(device)

    log: list[dict] = []
    data_iter = build_data_iter("train", block_size, batch_size, HF_CACHE)

    t0 = time.time()
    step = start_step

    while step < max_steps:
        # LR schedule
        lr = get_lr(step, warmup_steps, max_steps, cfg["max_lr"], cfg.get("min_lr", 1e-5))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        try:
            x, y = next(data_iter)
        except StopIteration:
            # Reset iterator — streaming datasets cycle via re-instantiation
            data_iter = build_data_iter("train", block_size, batch_size, HF_CACHE)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

        logits  = model(x)
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        if quant_fn is not None:
            ql          = quant_fn(model, current_step=step)
            total_loss  = ce_loss + ql
        else:
            ql         = torch.tensor(0.0)
            total_loss = ce_loss

        optimizer.zero_grad()
        total_loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        step += 1

        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - t0
            entry = {
                "step":      step,
                "lr":        lr,
                "ce_loss":   round(ce_loss.item(), 5),
                "quant_loss": round(ql.item(), 5),
                "elapsed_s": round(elapsed, 1),
            }
            log.append(entry)
            print(f"  step {step:>7}/{max_steps}  lr={lr:.2e}  "
                  f"CE={ce_loss.item():.4f}  QL={ql.item():.4f}  "
                  f"({elapsed:.0f}s elapsed)")

        if step % save_interval == 0 and log:
            save_checkpoint(model, optimizer, step, log, out_dir)

    # Final save
    save_checkpoint(model, optimizer, step, log, out_dir)
    print(f"\nTraining complete. Checkpoint: {out_dir / 'checkpoint_latest.pt'}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(cfg: dict) -> None:
    """Load log.json from each variant and overlay CE loss curves."""
    variants   = ["baseline", "topo_quant"]
    colors     = ["#757575", "#2196f3"]
    fig, ax    = plt.subplots(figsize=(9, 5))
    for variant, color in zip(variants, colors):
        log_path = OUTPUT_DIR / variant / "log.json"
        if not log_path.exists():
            continue
        with open(log_path) as f:
            log = json.load(f)
        steps = [e["step"]    for e in log]
        ces   = [e["ce_loss"] for e in log]
        ax.plot(steps, ces, label=variant, color=color, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("CE Loss")
    ax.set_title("GPT-2 Training Curves — Baseline vs TopoQuantLoss")
    ax.legend()
    plt.tight_layout()
    out = OUTPUT_DIR / "training_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Training curves saved to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GPT-2 small with TopoQuantLoss")
    p.add_argument("--config",    type=str,
                   default=str(BASE_DIR / "configs" / "topo_quant_gpt2.json"))
    p.add_argument("--variant",   type=str, default="topo_quant",
                   choices=["baseline", "topo_quant", "both"],
                   help="Which variant(s) to train")
    p.add_argument("--device",    type=str, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--plot-only", action="store_true",
                   help="Skip training; just regenerate the comparison plot")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    if args.device:    cfg["device"]    = args.device
    if args.max_steps: cfg["max_steps"] = args.max_steps

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        plot_training_curves(cfg)
        return

    variants_to_run = ["baseline", "topo_quant"] if args.variant == "both" else [args.variant]
    for v in variants_to_run:
        train(cfg, v)

    if len(variants_to_run) == 2:
        plot_training_curves(cfg)


if __name__ == "__main__":
    main()
