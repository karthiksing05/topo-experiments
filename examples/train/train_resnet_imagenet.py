"""
ResNet-18 trained on ImageNet with TopoLoss + per-channel entropy-sparsity penalty
applied to every convolutional layer in the four residual-block groups
(layer1 – layer4).  Optionally also covers the 1×1 downsample convolutions
(controlled by ``include_downsample`` in the config).

Three variants are trained in sequence and saved independently:
  topo       – TopoLoss (LaplacianPyramid) + entropy-sparsity penalty on every
               targeted residual-block conv.
  topo_only  – TopoLoss only; entropy penalty weights are zeroed.
  baseline   – Cross-entropy only; no topographic regularisation.

Training schedule follows the original He et al. (2015) ResNet paper:
  SGD + momentum, weight-decay 1e-4, LR=0.1 stepped by ×0.1 at epochs 30/60/80,
  90 total epochs, batch size 256.

Usage
-----
python examples/train/train_resnet_imagenet.py
python examples/train/train_resnet_imagenet.py --config configs/train_resnet_imagenet.json
python examples/train/train_resnet_imagenet.py --epochs 90 --device cuda:0 --data-dir /data/imagenet

The ImageNet root should contain ``train/`` and ``val/`` sub-directories in the
standard ImageFolder layout (one sub-folder per synset / class label).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
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
import torchvision.models as tv_models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from topoloss import LaplacianPyramid, TopoLoss
from topoloss.core import find_cortical_sheet_size

try:
    import datasets as hf_datasets          # huggingface/datasets
    _HF_DATASETS_AVAILABLE = True
except ImportError:
    _HF_DATASETS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs" / "train_resnet_imagenet"

# ---------------------------------------------------------------------------
# ImageNet-1k auto-download from Hugging Face Hub
# ---------------------------------------------------------------------------

def _dir_has_images(d: Path) -> bool:
    """True if *d* exists and contains at least one JPEG/PNG image file."""
    if not d.is_dir():
        return False
    for ext in ("*.JPEG", "*.jpeg", "*.jpg", "*.JPG", "*.png", "*.PNG"):
        if next(d.rglob(ext), None) is not None:
            return True
    return False


def maybe_download_imagenet(data_dir: Path, hf_token: str | None = None) -> None:
    """Download ImageNet-1k from Hugging Face Hub and write it as an ImageFolder tree.

    Only runs when *data_dir/train* or *data_dir/val* are absent / empty.
    The gated dataset ``ILSVRC/imagenet-1k`` on the Hugging Face Hub requires that you:
      1. Accept the terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k
      2. Pass your HF access token via *hf_token* (or set the ``HF_TOKEN``
         environment variable, or run ``huggingface-cli login`` beforehand).

    Layout written to disk::

        <data_dir>/
          train/<label_id: zero-padded 4-digit int>/<idx>.JPEG
          val/<label_id: zero-padded 4-digit int>/<idx>.JPEG

    Parameters
    ----------
    data_dir : root directory that will contain ``train/`` and ``val/``.
    hf_token : Hugging Face access token.  ``None`` falls back to the
               ``HF_TOKEN`` environment variable or a cached login.
    """
    import os

    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"

    need_train = not _dir_has_images(train_dir)
    need_val   = not _dir_has_images(val_dir)

    if not need_train and not need_val:
        return   # data already present

    if not _HF_DATASETS_AVAILABLE:
        raise ImportError(
            "The 'datasets' package is required for automatic ImageNet download.\n"
            "Install it with:  pip install datasets"
        )

    token = hf_token or os.environ.get("HF_TOKEN") or True  # True = cached creds

    print("\n" + "=" * 70)
    print("  IMAGENET-1k AUTO-DOWNLOAD")
    print("=" * 70)
    print("  Source  : huggingface.co/datasets/ILSVRC/imagenet-1k")
    print(f"  Target  : {data_dir}")
    print("  WARNING : Full ImageNet-1k is ~155 GB compressed.")
    print("            Make sure you have enough disk space and have accepted")
    print("            the dataset terms on Hugging Face Hub.")
    print()

    splits_needed: list[tuple[str, Path]] = []
    if need_train:
        splits_needed.append(("train", train_dir))
    if need_val:
        splits_needed.append(("validation", val_dir))  # HF uses 'validation'

    for hf_split, out_dir in splits_needed:
        print(f"  Downloading split '{hf_split}' -> {out_dir} ...")
        out_dir.mkdir(parents=True, exist_ok=True)

        ds = hf_datasets.load_dataset(
            "ILSVRC/imagenet-1k",
            split=hf_split,
            token=token,
            trust_remote_code=True,
        )

        n_total   = len(ds)
        n_written = 0
        n_skipped = 0
        t_start   = time.time()

        for idx, example in enumerate(ds):
            label    = example["label"]                      # int 0-999
            cls_dir  = out_dir / f"{label:04d}"
            cls_dir.mkdir(exist_ok=True)
            img_path = cls_dir / f"{idx:07d}.JPEG"

            if img_path.exists():                            # resume support
                n_skipped += 1
            else:
                img = example["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(img_path, format="JPEG", quality=95)
                n_written += 1

            if (idx + 1) % 10_000 == 0 or (idx + 1) == n_total:
                elapsed = time.time() - t_start
                pct     = 100 * (idx + 1) / n_total
                rate    = (n_written + n_skipped) / max(elapsed, 1e-3)
                eta     = (n_total - idx - 1) / max(rate, 1e-3)
                print(
                    f"    [{hf_split}] {idx+1:>7}/{n_total}  "
                    f"({pct:5.1f}%)  "
                    f"written={n_written}  skipped={n_skipped}  "
                    f"rate={rate:.0f} img/s  ETA={eta/60:.1f} min"
                )

        print(f"  Split '{hf_split}' done — {n_written} written, {n_skipped} skipped.\n")

    print("  ImageNet-1k download complete.")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Residual-block conv identification
# ---------------------------------------------------------------------------

_RESIDUAL_GROUPS = ("layer1", "layer2", "layer3", "layer4")


def _is_residual_conv(name: str, include_downsample: bool) -> bool:
    """True if *name* identifies a conv that should receive TopoLoss+entropy."""
    if not any(name.startswith(g) for g in _RESIDUAL_GROUPS):
        return False
    if "downsample" in name and not include_downsample:
        return False
    return True


def get_residual_convs(model: nn.Module,
                       include_downsample: bool = False) -> dict[str, nn.Conv2d]:
    """Return {module_name: Conv2d} for every targeted residual-block conv."""
    result: dict[str, nn.Conv2d] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and _is_residual_conv(name, include_downsample):
            result[name] = module
    return result


# ---------------------------------------------------------------------------
# Entropy-sparsity loss (re-used from train_topo_sparsity.py)
# ---------------------------------------------------------------------------

def cortical_entropy_loss(activations: torch.Tensor,
                          factor_h: float,
                          factor_w: float,
                          temperature: float = 1.0) -> torch.Tensor:
    """Per-sample entropy on the downsampled cortical sheet.

    For conv activations (B, C, H_feat, W_feat) we global-avg-pool over the
    spatial axes so that *output channels* form the topographic axis, giving a
    (B, C) tensor. That is then reshaped into the cortical sheet, downsampled,
    and converted to a probability distribution via softmax before entropy is
    computed.

    Returns the mean entropy over the batch (scalar, lower = sparser).
    """
    if activations.ndim == 4:
        activations = activations.mean(dim=(2, 3))   # (B, C)

    B, N  = activations.shape
    size  = find_cortical_sheet_size(N)
    H, W  = size.height, size.width

    sheet  = activations[:, :H * W].reshape(B, 1, H, W)
    H_d    = max(1, round(H / factor_h))
    W_d    = max(1, round(W / factor_w))
    flat   = F.adaptive_avg_pool2d(sheet, (H_d, W_d)).reshape(B, -1)

    probs  = F.softmax(flat / temperature, dim=-1)
    # entropy = -Σ p log p  — we want to *minimise* this (maximise sparsity)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
    return entropy


# ---------------------------------------------------------------------------
# Build TopoLoss for the targeted conv layers
# ---------------------------------------------------------------------------

def build_topo_loss(model: nn.Module,
                    residual_convs: dict[str, nn.Conv2d],
                    layer_cfg: dict,
                    default_cfg: dict) -> TopoLoss:
    """Construct a ``TopoLoss`` with one ``LaplacianPyramid`` per targeted conv.

    Parameters
    ----------
    model          : the ResNet-18 model (needed by ``LaplacianPyramid.from_layer``).
    residual_convs : {name: Conv2d} mapping produced by ``get_residual_convs``.
    layer_cfg      : per-layer config dict (from JSON ``layers`` field).
    default_cfg    : fallback config used for layers not listed in ``layer_cfg``.
    """
    pyramids = []
    for name, conv in residual_convs.items():
        lc = {**default_cfg, **layer_cfg.get(name, {})}
        pyramids.append(
            LaplacianPyramid.from_layer(
                model=model,
                layer=conv,
                factor_h=lc["factor_h"],
                factor_w=lc["factor_w"],
                scale=lc["topo_scale"],
            )
        )
    return TopoLoss(losses=pyramids)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(
    label: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epochs: int,
    ckpt_dir: Path,
    device: str,
    print_freq: int,
    save_freq: int,
    topo_loss: TopoLoss | None,
    residual_convs: dict[str, nn.Conv2d],
    layer_cfg: dict,
    default_layer_cfg: dict,
    start_epoch: int = 0,
    best_acc1: float = 0.0,
) -> nn.Module:
    """Train one model variant; return model loaded with best-checkpoint weights."""

    # ---- Register forward hooks to capture conv activations -----------------
    act_store: dict[str, torch.Tensor | None] = {n: None for n in residual_convs}
    hook_handles = []

    if topo_loss is not None:
        for name in residual_convs:
            def _make_hook(n: str):
                def _hook(_mod, _inp, out: torch.Tensor):
                    act_store[n] = out
                return _hook
            hook_handles.append(
                residual_convs[name].register_forward_hook(_make_hook(name))
            )

    # ---- Epoch loop ---------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        model.train()

        sum_ce        = 0.0
        sum_topo      = 0.0
        sum_entropy   = 0.0
        n_correct = n_total = 0
        t0 = time.time()

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            ce     = criterion(logits, labels)

            extra = torch.zeros(1, device=device)

            if topo_loss is not None:
                topo   = topo_loss.compute(model=model, reduce_mean=True)
                extra  = extra + topo
                sum_topo += topo.item() * imgs.size(0)

                for name, conv in residual_convs.items():
                    act = act_store[name]
                    if act is not None:
                        lc  = {**default_layer_cfg, **layer_cfg.get(name, {})}
                        lam = lc.get("lambda_entropy", 0.0)
                        if lam > 0.0:
                            ent   = cortical_entropy_loss(
                                act, lc["factor_h"], lc["factor_w"],
                                lc.get("temperature", 1.0),
                            )
                            extra = extra + lam * ent
                            sum_entropy += ent.item() * imgs.size(0)

            loss = ce + extra
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            sum_ce    += ce.item() * bs
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_total   += bs

            if print_freq > 0 and (batch_idx + 1) % print_freq == 0:
                acc = 100 * n_correct / n_total
                print(
                    f"[{label}] Epoch [{epoch+1}/{epochs}]  "
                    f"Step [{batch_idx+1}/{len(train_loader)}]  "
                    f"CE={sum_ce/n_total:.4f}  "
                    f"Topo={sum_topo/n_total:.5f}  "
                    f"Ent={sum_entropy/n_total:.5f}  "
                    f"Train@1={acc:.2f}%"
                )

        # ---- Scheduler step -------------------------------------------------
        scheduler.step()

        # ---- Validation -----------------------------------------------------
        top1, top5 = evaluate(model, val_loader, device)
        elapsed    = time.time() - t0
        lr_now     = optimizer.param_groups[0]["lr"]

        print(
            f"[{label}] Epoch [{epoch+1:3d}/{epochs}]  "
            f"CE={sum_ce/n_total:.4f}  "
            f"\033[93mTopo={sum_topo/n_total:.5f}  "
            f"Ent={sum_entropy/n_total:.5f}\033[0m  "
            f"Train@1={100*n_correct/n_total:.2f}%  "
            f"\033[92mVal@1={top1:.2f}%  Val@5={top5:.2f}%\033[0m  "
            f"LR={lr_now:.6f}  t={elapsed:.1f}s"
        )

        # ---- Checkpoint (every save_freq epochs + whenever best improves) ---
        is_best = top1 > best_acc1
        if is_best:
            best_acc1 = top1

        ckpt = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc1": best_acc1,
        }

        if is_best:
            torch.save(ckpt, ckpt_dir / f"best_{label}.pt")
            print(f"  *** New best Val@1 = {best_acc1:.2f}% — checkpoint saved ***")

        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            torch.save(ckpt, ckpt_dir / f"epoch{epoch+1:04d}_{label}.pt")

    # Last epoch always saved
    torch.save(ckpt, ckpt_dir / f"last_{label}.pt")

    for h in hook_handles:
        h.remove()

    print(f"\n[{label}] Training complete.  Best Val@1: {best_acc1:.2f}%")

    # Load best weights before returning
    best_path = ckpt_dir / f"best_{label}.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model"])
    return model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: str) -> tuple[float, float]:
    """Return (top-1 %, top-5 %) accuracy on *loader*."""
    model.eval()
    correct1 = correct5 = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)

        # Top-1
        pred1  = logits.argmax(dim=1)
        correct1 += (pred1 == labels).sum().item()

        # Top-5
        _, top5_preds = logits.topk(5, dim=1)
        correct5 += top5_preds.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

        total += labels.size(0)

    model.train()
    return 100 * correct1 / total, 100 * correct5 / total


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict = {
    "data_dir":    None,
    "output_dir":  None,
    "num_classes": 1000,

    "epochs":        90,
    "batch_size":    256,
    "lr":            0.1,
    "momentum":      0.9,
    "weight_decay":  1e-4,
    "lr_milestones": [30, 60, 80],
    "lr_gamma":      0.1,

    "device":      "cuda:0",
    "num_workers": 8,
    "print_freq":  200,
    "save_freq":   5,

    "resume_topo":      None,
    "resume_topo_only": None,
    "resume_base":      None,

    "include_downsample": False,

    "default_layer_cfg": {
        "topo_scale":     1.0,
        "factor_h":       2.0,
        "factor_w":       2.0,
        "lambda_entropy": 0.05,
        "temperature":    1.0,
    },

    "layers": {},
}


def get_config() -> dict:
    """Parse CLI; load JSON config; apply CLI overrides; return merged dict."""
    p = argparse.ArgumentParser(
        description="Train ResNet-18 on ImageNet with TopoLoss + entropy-sparsity penalty.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_cfg_path = str(BASE_DIR / "configs" / "train_resnet_imagenet.json")
    p.add_argument("--config",           type=str, default=default_cfg_path)
    p.add_argument("--data-dir",         type=str, default=None,
                   help="ImageNet root (must contain train/ and val/ subdirs)")
    p.add_argument("--output-dir",       type=str, default=None)
    p.add_argument("--num-classes",      type=int, default=None,
                   help="Number of output classes (1000 for full ImageNet)")
    p.add_argument("--epochs",           type=int, default=None)
    p.add_argument("--batch-size",       type=int, default=None)
    p.add_argument("--lr",               type=float, default=None)
    p.add_argument("--momentum",         type=float, default=None)
    p.add_argument("--weight-decay",     type=float, default=None)
    p.add_argument("--device",           type=str, default=None)
    p.add_argument("--num-workers",      type=int, default=None)
    p.add_argument("--print-freq",       type=int, default=None)
    p.add_argument("--save-freq",        type=int, default=None)
    p.add_argument("--resume-topo",      type=str, default=None)
    p.add_argument("--resume-topo-only", type=str, default=None)
    p.add_argument("--resume-base",      type=str, default=None)
    p.add_argument("--include-downsample", action="store_true", default=None,
                   help="Also apply TopoLoss+entropy to 1×1 downsample convs")
    p.add_argument("--hf-token", type=str, default=None,
                   help="Hugging Face access token for ImageNet-1k download "
                        "(falls back to HF_TOKEN env var or cached login)")
    cli = p.parse_args()

    cfg = copy.deepcopy(_DEFAULT_CONFIG)

    cfg_path = Path(cli.config)
    if cfg_path.exists():
        with open(cfg_path) as fh:
            file_cfg = json.load(fh)
        for key, val in file_cfg.items():
            if key.startswith("_comment"):
                continue
            if key == "layers" and isinstance(val, dict):
                for lname, lvals in val.items():
                    cfg["layers"][lname] = {**cfg["layers"].get(lname, {}), **lvals}
            elif key == "default_layer_cfg" and isinstance(val, dict):
                cfg["default_layer_cfg"].update(val)
            else:
                cfg[key] = val
        print(f"Config loaded from: {cfg_path}")
    else:
        print(f"Config file not found ({cfg_path}), using built-in defaults.")

    # CLI overrides
    cli_map = {
        "data_dir":          cli.data_dir,
        "output_dir":        cli.output_dir,
        "num_classes":       cli.num_classes,
        "epochs":            cli.epochs,
        "batch_size":        cli.batch_size,
        "lr":                cli.lr,
        "momentum":          cli.momentum,
        "weight_decay":      cli.weight_decay,
        "device":            cli.device,
        "num_workers":       cli.num_workers,
        "print_freq":        cli.print_freq,
        "save_freq":         cli.save_freq,
        "resume_topo":       cli.resume_topo,
        "resume_topo_only":  cli.resume_topo_only,
        "resume_base":       cli.resume_base,
    }
    for key, val in cli_map.items():
        if val is not None:
            cfg[key] = val
    if cli.include_downsample:
        cfg["include_downsample"] = True
    if cli.hf_token is not None:
        cfg["hf_token"] = cli.hf_token

    if cfg["data_dir"] is None:
        cfg["data_dir"] = str(BASE_DIR / "data" / "imagenet")
    if cfg["output_dir"] is None:
        cfg["output_dir"] = str(OUTPUT_DIR)

    cfg.setdefault("hf_token", None)
    return cfg


# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    # ---- Print resolved config ----------------------------------------------
    print("=" * 70)
    print("  EFFECTIVE CONFIG")
    print("=" * 70)
    skip_keys = {"layers", "default_layer_cfg"}
    print(json.dumps({k: v for k, v in cfg.items() if k not in skip_keys}, indent=2))
    print("  default_layer_cfg:", json.dumps(cfg["default_layer_cfg"]))
    print(f"  layers: {len(cfg['layers'])} per-layer overrides")
    print()

    # ---- Device setup -------------------------------------------------------
    if torch.cuda.is_available():
        device = cfg["device"]
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}\n")

    out_dir  = Path(cfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data loading (auto-download if needed) ----------------------------
    data_dir  = Path(cfg["data_dir"])
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"

    if not _dir_has_images(train_dir) or not _dir_has_images(val_dir):
        print(f"ImageNet not found at {data_dir} — attempting auto-download ...")
        maybe_download_imagenet(data_dir, hf_token=cfg.get("hf_token"))

    # Verify again after attempted download
    if not _dir_has_images(train_dir) or not _dir_has_images(val_dir):
        raise FileNotFoundError(
            f"ImageNet train/val directories still empty after download attempt.\n"
            f"Expected layout:\n"
            f"  {train_dir}/<label>/*.JPEG\n"
            f"  {val_dir}/<label>/*.JPEG\n"
            "If this is a gated dataset, make sure you have:\n"
            "  1. Accepted terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k\n"
            "  2. Provided a valid token via --hf-token or the HF_TOKEN env var."
        )

    # Standard ImageNet augmentation (He et al. 2015 / torchvision convention)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_ds   = datasets.ImageFolder(str(val_dir),   transform=val_transform)

    _pin = str(device).startswith("cuda")
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=_pin,
        persistent_workers=(cfg["num_workers"] > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=_pin,
        persistent_workers=(cfg["num_workers"] > 0),
    )
    print(f"Dataset: {len(train_ds):,} train | {len(val_ds):,} val | "
          f"{cfg['num_classes']} classes\n")

    layer_cfg       = cfg["layers"]
    default_lc      = cfg["default_layer_cfg"]
    incl_down       = cfg["include_downsample"]

    def _make_model() -> nn.Module:
        """Create a fresh ResNet-18 adapted for num_classes."""
        m = tv_models.resnet18(weights=None, num_classes=cfg["num_classes"])
        return m.to(device)

    def _make_optimizer(model: nn.Module) -> tuple:
        opt = optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"],
        )
        sched = optim.lr_scheduler.MultiStepLR(
            opt, milestones=cfg["lr_milestones"], gamma=cfg["lr_gamma"]
        )
        return opt, sched

    def _resume(model, opt, sched, ckpt_path):
        if not ckpt_path:
            return 0, 0.0
        p = Path(ckpt_path)
        if not p.exists():
            print(f"  Resume checkpoint not found: {p}")
            return 0, 0.0
        ckpt = torch.load(p, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
        start  = ckpt["epoch"] + 1
        best   = ckpt.get("best_acc1", 0.0)
        print(f"  Resumed from epoch {start}  (best_acc1={best:.2f}%)")
        return start, best

    criterion = nn.CrossEntropyLoss().to(device)

    # =========================================================================
    # 1. Topo model  (TopoLoss + entropy penalty)
    # =========================================================================
    print("=" * 70)
    print("  TOPO MODEL  (TopoLoss + entropy-sparsity on all residual-block convs)")
    print("=" * 70)
    topo_model = _make_model()
    topo_opt, topo_sched = _make_optimizer(topo_model)
    topo_start, topo_best = _resume(topo_model, topo_opt, topo_sched,
                                    cfg.get("resume_topo"))

    topo_convs = get_residual_convs(topo_model, include_downsample=incl_down)
    print(f"  Targeting {len(topo_convs)} conv layers: {list(topo_convs.keys())}\n")

    topo_loss_fn = build_topo_loss(topo_model, topo_convs, layer_cfg, default_lc)

    topo_model = run_training(
        label="topo",
        model=topo_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=topo_opt,
        scheduler=topo_sched,
        epochs=cfg["epochs"],
        ckpt_dir=ckpt_dir,
        device=device,
        print_freq=cfg["print_freq"],
        save_freq=cfg["save_freq"],
        topo_loss=topo_loss_fn,
        residual_convs=topo_convs,
        layer_cfg=layer_cfg,
        default_layer_cfg=default_lc,
        start_epoch=topo_start,
        best_acc1=topo_best,
    )

    # =========================================================================
    # 2. Topo-only model  (TopoLoss, no entropy)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  TOPO-ONLY MODEL  (TopoLoss only — entropy penalty zeroed)")
    print("=" * 70)
    topo_only_model = _make_model()
    topo_only_opt, topo_only_sched = _make_optimizer(topo_only_model)
    topo_only_start, topo_only_best = _resume(topo_only_model, topo_only_opt,
                                              topo_only_sched,
                                              cfg.get("resume_topo_only"))

    topo_only_convs = get_residual_convs(topo_only_model, include_downsample=incl_down)

    # Zero out entropy weights for the topo-only variant
    topo_only_layer_cfg = {
        name: {**vals, "lambda_entropy": 0.0}
        for name, vals in layer_cfg.items()
    }
    topo_only_default_lc = {**default_lc, "lambda_entropy": 0.0}

    topo_only_loss_fn = build_topo_loss(
        topo_only_model, topo_only_convs, layer_cfg, default_lc
    )

    topo_only_model = run_training(
        label="topo_only",
        model=topo_only_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=topo_only_opt,
        scheduler=topo_only_sched,
        epochs=cfg["epochs"],
        ckpt_dir=ckpt_dir,
        device=device,
        print_freq=cfg["print_freq"],
        save_freq=cfg["save_freq"],
        topo_loss=topo_only_loss_fn,
        residual_convs=topo_only_convs,
        layer_cfg=topo_only_layer_cfg,
        default_layer_cfg=topo_only_default_lc,
        start_epoch=topo_only_start,
        best_acc1=topo_only_best,
    )

    # =========================================================================
    # 3. Baseline model  (CE only)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  BASELINE MODEL  (CrossEntropy only — no TopoLoss, no entropy)")
    print("=" * 70)
    base_model = _make_model()
    base_opt, base_sched = _make_optimizer(base_model)
    base_start, base_best = _resume(base_model, base_opt, base_sched,
                                    cfg.get("resume_base"))

    base_model = run_training(
        label="baseline",
        model=base_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=base_opt,
        scheduler=base_sched,
        epochs=cfg["epochs"],
        ckpt_dir=ckpt_dir,
        device=device,
        print_freq=cfg["print_freq"],
        save_freq=cfg["save_freq"],
        topo_loss=None,
        residual_convs={},
        layer_cfg={},
        default_layer_cfg=default_lc,
        start_epoch=base_start,
        best_acc1=base_best,
    )

    print(f"\nAll models trained.  Checkpoints in: {ckpt_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = get_config()

    # Tee stdout/stderr to a timestamped log file
    log_dir  = BASE_DIR / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"train_resnet_imagenet-{ts}.txt"

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data: str) -> None:
            for s in self._streams:
                try:
                    s.write(data)
                except Exception:
                    pass

        def flush(self) -> None:
            for s in self._streams:
                try:
                    s.flush()
                except Exception:
                    pass

        def isatty(self) -> bool:
            return any(hasattr(s, "isatty") and s.isatty() for s in self._streams)

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    with open(log_path, "w") as fh:
        sys.stdout = _Tee(orig_stdout, fh)
        sys.stderr = _Tee(orig_stderr, fh)
        try:
            train(cfg)
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

    print(f"Saved run log -> {log_path}")
