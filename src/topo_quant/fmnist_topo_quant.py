"""fmnist_topo_quant.py
======================
Preliminary test of TopoQuantLoss on Fashion-MNIST.

Trains four variants on the *same* SimpleNN architecture used in the forgetting
experiments and reports, for each variant:
  • Final test accuracy (%)
  • Number of trainable parameters
  • In-memory model size (MB)  — computed via parameter byte count
  • Effective 4-bit quantised size (MB)  — theoretical post-quantisation size

Variants
--------
  baseline             — CrossEntropy only
  topo_quant           — CrossEntropy + TopoQuantLoss (hard STE)
  soft_topo_quant      — CrossEntropy + SoftTopoQuantLoss (cosine-annealed temperature)
  combined_topo_quant  — CrossEntropy + CombinedTopoQuantLoss (spatial + quant)

Outputs saved to:
  outputs/topo_quant_fmnist/results_latest.json
  outputs/topo_quant_fmnist/results_{timestamp}.json
  outputs/topo_quant_fmnist/accuracy_comparison.png

Usage
-----
    python src/topo_quant/fmnist_topo_quant.py [--config configs/topo_quant_fmnist.json]
                                               [--device cuda:0]
                                               [--epochs 20]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Repo-local imports
BASE_DIR   = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs" / "topo_quant_fmnist"
sys.path.insert(0, str(BASE_DIR / "src" / "topo_quant"))

from topo_quant_loss import TopoQuantLoss, SoftTopoQuantLoss, CombinedTopoQuantLoss

# ---------------------------------------------------------------------------
# Model (identical to fmnist_forgetting SimpleNN — relu baseline mode)
# ---------------------------------------------------------------------------

class SimpleNN(nn.Module):
    """Two-layer MLP matching the forgetting-experiment architecture."""

    def __init__(self, hidden_size: int = 256, bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, 10,       bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28 * 28)
        return self.fc2(F.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Model sizing utilities
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module, bits_per_param: int = 32) -> float:
    """Theoretical model size in MB at a given bits-per-param."""
    n = count_params(model)
    return n * bits_per_param / 8 / (1024 ** 2)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    quant_loss_fn,
    device: torch.device,
    global_step: int,
) -> tuple[float, float, int]:
    """Returns (avg_ce_loss, avg_quant_loss, updated_global_step)."""
    model.train()
    total_ce = 0.0
    total_ql = 0.0
    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)

        logits   = model(inputs)
        ce_loss  = criterion(logits, targets)

        if quant_loss_fn is not None:
            if isinstance(quant_loss_fn, SoftTopoQuantLoss):
                ql = quant_loss_fn(model, current_step=global_step)
            elif isinstance(quant_loss_fn, CombinedTopoQuantLoss):
                ql = quant_loss_fn(model, current_step=global_step)
            else:
                ql = quant_loss_fn(model)
            total_loss = ce_loss + ql
        else:
            ql         = torch.tensor(0.0)
            total_loss = ce_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_ce += ce_loss.item()
        total_ql += ql.item()
        global_step += 1

    n = len(loader)
    return total_ce / n, total_ql / n, global_step


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total   = 0
    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        preds   = model(inputs).argmax(dim=1)
        correct += (preds == targets).sum().item()
        total   += targets.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_quantized(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_bits: int,
) -> float:
    """Evaluate model after applying hard uniform quantization to all weights.

    Quantization is applied in-place on a deep copy so the original model
    is not modified.
    """
    import copy
    from topo_quant_loss import _uniform_quantise

    q_model = copy.deepcopy(model)
    q_model.eval()
    with torch.no_grad():
        for param in q_model.parameters():
            if param.ndim >= 1:
                param.copy_(_uniform_quantise(param.data, num_bits).detach())

    correct = 0
    total   = 0
    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        preds   = q_model(inputs).argmax(dim=1)
        correct += (preds == targets).sum().item()
        total   += targets.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(cfg: dict) -> dict:
    device_str = cfg.get("device", "cuda:0")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{device_str}' but torch.cuda.is_available() is False. "
            "Check that the correct conda environment is active and the CUDA driver is "
            "accessible on this node (nvidia-smi, CUDA_VISIBLE_DEVICES)."
        )
    device = torch.device(device_str)
    print(f"Device: {device}")

    # -- Data -----------------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    data_dir = cfg.get("data_dir") or str(BASE_DIR / "data")
    train_ds = datasets.FashionMNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds  = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)

    epochs      = cfg["epochs"]
    hidden_size = cfg["hidden_size"]
    lr          = cfg["lr"]
    num_bits    = cfg["num_bits"]
    criterion   = nn.CrossEntropyLoss()

    # Build loss functions
    variant_configs = {
        "baseline": None,
        "topo_quant": TopoQuantLoss(
            num_bits  = num_bits,
            tau       = cfg["tau"],
        ),
        "soft_topo_quant": SoftTopoQuantLoss(
            num_bits            = num_bits,
            tau                 = cfg["tau"],
            initial_temperature = cfg["soft_initial_temp"],
            final_temperature   = cfg["soft_final_temp"],
            anneal_steps        = epochs * len(train_loader),
            anneal_schedule     = cfg.get("anneal_schedule", "cosine"),
        ),
        "combined_topo_quant": CombinedTopoQuantLoss(
            num_bits                  = num_bits,
            tau_spatial               = cfg["tau_spatial"],
            tau_precision             = cfg["tau"],
            spatial_downsample_factor = cfg.get("spatial_downsample_factor", 3),
        ),
    }

    results = {}
    for variant, quant_fn in variant_configs.items():
        print(f"\n{'='*60}")
        print(f"Variant: {variant}")
        print(f"{'='*60}")

        model     = SimpleNN(hidden_size=hidden_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if quant_fn is not None:
            quant_fn = quant_fn.to(device)

        global_step   = 0
        epoch_history = []

        for epoch in range(1, epochs + 1):
            ce, ql, global_step = train_one_epoch(
                model, train_loader, criterion, optimizer,
                quant_fn, device, global_step,
            )
            val_acc = evaluate(model, test_loader, device)
            epoch_history.append({"epoch": epoch, "ce": ce, "quant_loss": ql, "val_acc": val_acc})
            if epoch % cfg.get("print_freq", 5) == 0 or epoch == epochs:
                print(f"  Epoch {epoch:>3}/{epochs}  CE={ce:.4f}  QL={ql:.4f}  Acc={val_acc:.2f}%")

        final_acc    = epoch_history[-1]["val_acc"]
        print(f"  Evaluating quantized model ({num_bits}-bit) ...")
        quant_acc    = evaluate_quantized(model, test_loader, device, num_bits)
        acc_drop     = round(final_acc - quant_acc, 3)
        n_params     = count_params(model)
        fp32_mb      = model_size_mb(model, bits_per_param=32)
        quant_mb     = model_size_mb(model, bits_per_param=num_bits)

        results[variant] = {
            "fp32_accuracy":      round(final_acc, 3),
            "quantized_accuracy": round(quant_acc, 3),
            "accuracy_drop":      acc_drop,
            "n_params":           n_params,
            "fp32_size_mb":       round(fp32_mb, 4),
            "quantised_size_mb":  round(quant_mb, 4),
            "num_bits":           num_bits,
            "epochs":             epoch_history,
        }
        print(f"\n  FP32 accuracy       : {final_acc:.2f}%")
        print(f"  {num_bits}-bit quant accuracy  : {quant_acc:.2f}%")
        print(f"  Accuracy drop       : {acc_drop:.3f}pp")
        print(f"  Parameters          : {n_params:,}")
        print(f"  FP32 size           : {fp32_mb:.4f} MB")
        print(f"  {num_bits}-bit size           : {quant_mb:.4f} MB")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: dict, output_dir: Path, num_bits: int) -> None:
    variants   = list(results.keys())
    colors     = ["#757575", "#2196f3", "#4caf50", "#ff5722"]
    fp32_accs  = [results[v]["fp32_accuracy"]      for v in variants]
    quant_accs = [results[v]["quantized_accuracy"]  for v in variants]
    drops      = [results[v]["accuracy_drop"]       for v in variants]

    x      = np.arange(len(variants))
    width  = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 1: FP32 vs quantized accuracy (grouped bar)
    ax = axes[0]
    b1 = ax.bar(x - width / 2, fp32_accs,  width, label="FP32",
                color=colors[:len(variants)], edgecolor="black", linewidth=0.7)
    b2 = ax.bar(x + width / 2, quant_accs, width, label=f"{num_bits}-bit quant",
                color=colors[:len(variants)], edgecolor="black", linewidth=0.7,
                alpha=0.45, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"FP32 vs {num_bits}-bit Accuracy")
    lo = max(0, min(fp32_accs + quant_accs) - 3)
    ax.set_ylim(lo, 100)
    ax.legend(fontsize=8)
    for bar, acc in zip(b1, fp32_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{acc:.1f}", ha="center", va="bottom", fontsize=7)
    for bar, acc in zip(b2, quant_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{acc:.1f}", ha="center", va="bottom", fontsize=7)

    # Panel 2: accuracy drop bar chart
    ax2 = axes[1]
    drop_colors = [c if d >= 0 else "#e53935" for c, d in zip(colors, drops)]
    bars2 = ax2.bar(variants, drops, color=drop_colors, edgecolor="black", linewidth=0.7)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Accuracy Drop (pp)")
    ax2.set_title(f"FP32 → {num_bits}-bit Accuracy Drop (lower = better)")
    ax2.tick_params(axis="x", rotation=15)
    ax2.set_xticklabels(variants, rotation=15, ha="right", fontsize=8)
    for bar, d in zip(bars2, drops):
        va   = "bottom" if d >= 0 else "top"
        yoff = 0.02 if d >= 0 else -0.02
        ax2.text(bar.get_x() + bar.get_width() / 2, d + yoff,
                 f"{d:.2f}pp", ha="center", va=va, fontsize=8)

    # Panel 3: val-accuracy training curves
    ax3 = axes[2]
    for v, color in zip(variants, colors):
        hist = results[v]["epochs"]
        xs = [e["epoch"]   for e in hist]
        ys = [e["val_acc"] for e in hist]
        ax3.plot(xs, ys, label=v, color=color, marker="o", markersize=3)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Val Accuracy (%)")
    ax3.set_title("Training Curves")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    out_path = output_dir / "accuracy_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: dict, num_bits: int) -> None:
    w = 90
    print(f"\n{'='*w}")
    print(f"{'Variant':<25} {'FP32 Acc':>9} {'Q Acc':>9} {'Drop (pp)':>10} "
          f"{'Params':>10} {'FP32 MB':>9} {f'{num_bits}b MB':>9}")
    print(f"{'-'*w}")
    for variant, r in results.items():
        print(f"{variant:<25} {r['fp32_accuracy']:>9.2f} {r['quantized_accuracy']:>9.2f} "
              f"{r['accuracy_drop']:>10.3f} {r['n_params']:>10,} "
              f"{r['fp32_size_mb']:>9.4f} {r['quantised_size_mb']:>9.4f}")
    print(f"{'='*w}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TopoQuantLoss FashionMNIST benchmark")
    p.add_argument("--config", type=str,
                   default=str(BASE_DIR / "configs" / "topo_quant_fmnist.json"))
    p.add_argument("--device",  type=str, default=None)
    p.add_argument("--epochs",  type=int, default=None)
    p.add_argument("--num-bits", type=int, default=None)
    p.add_argument("--tau",     type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    # CLI overrides
    if args.device:   cfg["device"]   = args.device
    if args.epochs:   cfg["epochs"]   = args.epochs
    if args.num_bits: cfg["num_bits"] = args.num_bits
    if args.tau:      cfg["tau"]      = args.tau

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = run_experiment(cfg)

    print_summary(results, cfg["num_bits"])
    plot_results(results, OUTPUT_DIR, cfg["num_bits"])

    # Serialise: strip per-epoch history for the "latest" summary then save full
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path   = OUTPUT_DIR / f"results_{ts}.json"
    latest_path = OUTPUT_DIR / "results_latest.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    # Latest summary without per-epoch lists
    summary = {
        v: {k: val for k, val in r.items() if k != "epochs"}
        for v, r in results.items()
    }
    with open(latest_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFull results  : {full_path}")
    print(f"Latest summary: {latest_path}")


if __name__ == "__main__":
    main()
