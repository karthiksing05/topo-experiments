"""
Shared utilities for STL-10 → CIFAR-10 catastrophic forgetting experiments.

Experimental design
-------------------
Phase 1 — Pretrain on STL-10 (96×96, 10 classes, 5k train samples).
Phase 2 — Finetune on CIFAR-10 (resized to 96×96, 10 classes, 50k train samples).
Measure  — STL accuracy before/after finetuning, CIFAR accuracy after.
           Forgetting = STL_before − STL_after.

Three variants are compared:
  baseline        : CrossEntropy only
  topo_only       : CrossEntropy + TopoLoss (LaplacianPyramid)
  topo_sparsity   : CrossEntropy + TopoLoss + per-channel entropy-sparsity

Uses DataParallel + AMP automatically when multiple GPUs / CUDA is available.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as tv_models
import torchvision.transforms as T
from torch.utils.data import DataLoader

from topoloss import LaplacianPyramid, TopoLoss
from topoloss.core import find_cortical_sheet_size

# Pull residual-conv helpers and entropy loss from the ImageNet common module
# instead of duplicating them.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent / "imagenet"))  # resnet_imagenet_common lives there
from resnet_imagenet_common import (
    get_residual_convs,
    cortical_entropy_loss,
    build_topo_loss,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs" / "stl_cifar"

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def make_stl_loaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    img_size: int = 96,
) -> tuple[DataLoader, DataLoader]:
    """STL-10 train / test loaders (96×96 by default)."""
    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(img_size, padding=12),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    val_tf = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    train_ds = torchvision.datasets.STL10(
        root=data_dir, split="train", download=True, transform=train_tf
    )
    val_ds = torchvision.datasets.STL10(
        root=data_dir, split="test", download=True, transform=val_tf
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def make_cifar_loaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    img_size: int = 96,
) -> tuple[DataLoader, DataLoader]:
    """CIFAR-10 loaders with images resized to *img_size* (default 96)."""
    train_tf = T.Compose([
        T.Resize(img_size),
        T.RandomHorizontalFlip(),
        T.RandomCrop(img_size, padding=12),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    val_tf = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    val_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_tf
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# CIFAR-10 → STL-10 overlap loaders
# ---------------------------------------------------------------------------

# Overlapping classes between CIFAR-10 and STL-10.
# CIFAR automobile(1) is treated as equivalent to STL car(2).
# CIFAR frog(6) has no STL equivalent and is excluded.
_CIFAR_TO_STL: dict[int, int] = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 7: 6, 8: 8, 9: 9}


class _RemappedSubset(torch.utils.data.Dataset):
    """A dataset view that remaps labels via *label_map* and keeps only
    samples whose original label appears in *label_map*."""

    def __init__(
        self,
        base_ds: torch.utils.data.Dataset,
        indices: list[int],
        label_map: dict[int, int],
    ):
        self.base_ds   = base_ds
        self.indices   = indices
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, lbl = self.base_ds[self.indices[i]]
        return img, self.label_map[lbl]


def make_cifar_overlap_loaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    img_size: int = 96,
) -> tuple[DataLoader, DataLoader]:
    """CIFAR-10 loaders filtered to STL-10-overlapping classes.

    Labels are remapped to their STL-10 integer index so that a
    model trained on STL-10 can be fine-tuned without a head swap.
    CIFAR frog (label 6) is excluded; CIFAR automobile (label 1) is
    mapped to STL car (label 2).
    """
    train_tf = T.Compose([
        T.Resize(img_size),
        T.RandomHorizontalFlip(),
        T.RandomCrop(img_size, padding=12),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    val_tf = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=train_tf
    )
    val_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_tf
    )

    def _filter(ds):
        indices = [
            i for i, lbl in enumerate(ds.targets)
            if lbl in _CIFAR_TO_STL
        ]
        return _RemappedSubset(ds, indices, _CIFAR_TO_STL)

    train_sub = _filter(train_ds)
    val_sub   = _filter(val_ds)

    print(
        f"  CIFAR-10 overlap subset: "
        f"train {len(train_sub):,} / val {len(val_sub):,} samples "
        f"({len(_CIFAR_TO_STL)} of 10 classes, labels remapped to STL-10 space)"
    )

    train_loader = DataLoader(
        train_sub, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_sub, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Cortical sparsity losses (KL-from-uniform + per-sample entropy)
# ---------------------------------------------------------------------------

def cortical_kl_entropy_losses(
    activations: torch.Tensor,
    factor_h: float,
    factor_w: float,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-batch KL-from-uniform and per-sample entropy on
    the downsampled cortical sheet.

    Returns
    -------
    kl_loss : scalar tensor
        KL divergence of the batch-mean softmax from the uniform distribution.
    entropy_loss : scalar tensor
        Mean per-sample softmax entropy (encourages sparsity when minimised).
    """
    if activations.ndim == 4:
        activations = activations.mean(dim=(2, 3))
    B, N = activations.shape
    size = find_cortical_sheet_size(N)
    H, W = size.height, size.width
    sheet = activations[:, : H * W].reshape(B, 1, H, W)
    H_d = max(1, round(H / factor_h))
    W_d = max(1, round(W / factor_w))
    flat = F.adaptive_avg_pool2d(sheet, (H_d, W_d)).reshape(B, -1)
    M    = flat.shape[1]
    probs      = F.softmax(flat / temperature, dim=-1)          # (B, M)
    batch_mean = probs.mean(dim=0)                               # (M,)
    kl_loss = F.kl_div(
        torch.full_like(batch_mean, -math.log(M)),               # log-uniform target
        batch_mean,
        reduction="sum",
    )
    entropy_loss = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
    return kl_loss, entropy_loss


def cortical_l1_entropy_loss(
    activations: torch.Tensor,
    factor_h: float,
    factor_w: float,
) -> torch.Tensor:
    """Single sparsity penalty using L1-normalised activation entropy.

    Treats the magnitude of each cortical unit as a probability mass
    (|x_i| / sum|x|).  Minimising this entropy drives the sheet toward
    one-hot firing — exact zeros receive zero probability mass, so the
    model is directly rewarded for silencing units.

    Unlike softmax-based losses, zeros contribute nothing here: a truly
    silent unit is genuinely 'sparse'.  No temperature parameter needed.

    Returns
    -------
    scalar tensor — mean per-sample L1-entropy (lower = sparser).
    """
    if activations.ndim == 4:
        activations = activations.mean(dim=(2, 3))
    B, N = activations.shape
    size = find_cortical_sheet_size(N)
    H, W = size.height, size.width
    sheet = activations[:, : H * W].reshape(B, 1, H, W)
    H_d = max(1, round(H / factor_h))
    W_d = max(1, round(W / factor_w))
    flat = F.adaptive_avg_pool2d(sheet, (H_d, W_d)).reshape(B, -1)  # (B, M)
    mags = flat.abs()                                                # (B, M)
    norm = mags.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    p    = mags / norm                                               # (B, M)
    entropy = -(p * (p + 1e-10).log()).sum(dim=-1)                   # (B,)
    return entropy.mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 10, device: str = "cuda") -> nn.Module:
    """ResNet-18 with a fresh classification head for *num_classes*."""
    m = tv_models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(device)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return opt


def _grad_entropy(model: nn.Module) -> float:
    """Mean normalised Shannon entropy of gradient magnitudes across all layers.

    For each parameter tensor that has a gradient, we treat |grad| as an
    unnormalised probability distribution, normalise it to sum-1, compute
    Shannon entropy, then divide by log(n) to get a value in [0, 1].
    Returns the average of those per-layer entropies.

    Interpretation:
      1.0  — perfectly uniform gradient (every weight pulled equally)
      0.0  — all gradient mass concentrated on one weight
    """
    entropies: list[float] = []
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float().abs().view(-1)
        # Skip if AMP scaler left inf/nan gradients (skipped update step)
        if not torch.isfinite(g).all():
            continue
        total = g.sum()
        if total == 0 or g.numel() < 2:
            continue
        prob = g / total
        eps  = 1e-10
        h    = -(prob * (prob + eps).log()).sum().item()
        entropies.append(h / math.log(g.numel()))
    return float(sum(entropies) / len(entropies)) if entropies else 0.0


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Return top-1 accuracy (%) on *loader*."""
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    model.train()
    return 100.0 * correct / total


def _train_one_epoch(
    label: str,
    epoch: int,
    total_epochs: int,
    phase: str,
    train_model: nn.Module,
    base_model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler,              # GradScaler | None
    device: str,
    topo_loss: Optional[TopoLoss],
    residual_convs: dict[str, nn.Conv2d],
    act_store: dict[str, Optional[torch.Tensor]],
    layer_cfg: dict,
    default_layer_cfg: dict,
    print_freq: int,
) -> tuple[float, float, float, float, float, float]:
    """Run one epoch; return (ce_avg, topo_avg, entropy_avg, kl_avg, sparse_avg, grad_entropy_avg)."""
    train_model.train()
    sum_ce = sum_topo = sum_entropy = sum_kl = sum_sparse = sum_grad_entropy = 0.0
    n_steps = 0
    n_total = 0
    primary = torch.device(device)

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits = train_model(imgs)
            ce     = criterion(logits, labels)
            extra  = torch.zeros(1, device=imgs.device)

            if topo_loss is not None:
                topo  = topo_loss.compute(model=base_model, reduce_mean=True).to(primary)
                extra = extra + topo
                sum_topo += topo.item() * imgs.size(0)

                for name in residual_convs:
                    act = act_store.get(name)
                    if act is not None:
                        act = act.to(primary)
                        lc  = {**default_layer_cfg, **layer_cfg.get(name, {})}
                        lam_ent = lc.get("lambda_entropy", 0.0)
                        lam_kl  = lc.get("lambda_kl",      0.0)
                        if lam_ent > 0.0 or lam_kl > 0.0:
                            kl_val, ent = cortical_kl_entropy_losses(
                                act, lc["factor_h"], lc["factor_w"],
                                lc.get("temperature", 1.0),
                            )
                            if lam_ent > 0.0:
                                extra        = extra + lam_ent * ent
                                sum_entropy += ent.item() * imgs.size(0)
                            if lam_kl > 0.0:
                                extra   = extra + lam_kl * kl_val
                                sum_kl += kl_val.item() * imgs.size(0)
                        lam_sparse = lc.get("lambda_sparse", 0.0)
                        if lam_sparse > 0.0:
                            sparse_val  = cortical_l1_entropy_loss(
                                act, lc["factor_h"], lc["factor_w"],
                            )
                            extra       = extra + lam_sparse * sparse_val
                            sum_sparse += sparse_val.item() * imgs.size(0)

            loss = ce + extra

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            sum_grad_entropy += _grad_entropy(base_model)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            sum_grad_entropy += _grad_entropy(base_model)
            optimizer.step()

        bs       = imgs.size(0)
        sum_ce  += ce.item() * bs
        n_total += bs
        n_steps += 1

        if print_freq > 0 and (batch_idx + 1) % print_freq == 0:
            print(
                f"[{label}|{phase}] Epoch [{epoch+1}/{total_epochs}]  "
                f"Step [{batch_idx+1}/{len(loader)}]  "
                f"CE={sum_ce/n_total:.4f}  "
                f"Topo={sum_topo/n_total:.5f}  "
                f"Ent={sum_entropy/n_total:.5f}  "
                f"KL={sum_kl/n_total:.5f}  "
                f"Sparse={sum_sparse/n_total:.5f}  "
                f"GradH={sum_grad_entropy/max(n_steps,1):.4f}"
            )

    grad_ent_avg = sum_grad_entropy / max(n_steps, 1)
    return sum_ce / n_total, sum_topo / n_total, sum_entropy / n_total, sum_kl / n_total, sum_sparse / n_total, grad_ent_avg


def run_pretrain(
    label: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    ckpt_dir: Path,
    topo_loss: Optional[TopoLoss],
    residual_convs: dict[str, nn.Conv2d],
    layer_cfg: dict,
    default_layer_cfg: dict,
    start_epoch: int = 0,
    best_acc: float = 0.0,
) -> tuple[nn.Module, float, list[float]]:
    """Phase 1: train on STL-10.

    Returns (model_with_best_weights, best_stl_acc, acc_per_epoch).
    """
    device   = cfg["device"]
    epochs   = cfg["stl_epochs"]
    use_cuda = device.startswith("cuda")

    # DataParallel
    base_model  = model
    train_model = model
    num_gpus    = torch.cuda.device_count() if use_cuda else 0
    if num_gpus > 1:
        print(f"  DataParallel: using {num_gpus} GPUs")
        train_model = nn.DataParallel(model)

    scaler = torch.amp.GradScaler("cuda") if use_cuda else None
    if scaler:
        print("  AMP enabled")

    # Hooks for entropy loss
    act_store:   dict[str, Optional[torch.Tensor]] = {n: None for n in residual_convs}
    hook_handles = []
    if topo_loss is not None:
        for name in residual_convs:
            def _hook_fn(n: str):
                def _h(_m, _i, out):
                    act_store[n] = out
                return _h
            hook_handles.append(
                residual_convs[name].register_forward_hook(_hook_fn(name))
            )

    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(base_model, cfg["lr"], cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Restore from checkpoint if requested
    if cfg.get("resume_pretrain") and Path(cfg["resume_pretrain"]).exists():
        ckpt       = torch.load(cfg["resume_pretrain"], map_location=device, weights_only=False)
        base_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc    = ckpt.get("best_acc", 0.0)
        print(f"  Resumed pretrain from {cfg['resume_pretrain']} (epoch {start_epoch})")

    acc_history:      list[float] = []
    ce_history:       list[float] = []
    topo_history:     list[float] = []
    ent_history:      list[float] = []
    kl_history:       list[float] = []
    sparse_history:   list[float] = []
    grad_ent_history: list[float] = []

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        ce_avg, topo_avg, ent_avg, kl_avg, sparse_avg, grad_ent_avg = _train_one_epoch(
            label=label, epoch=epoch, total_epochs=epochs, phase="STL",
            train_model=train_model, base_model=base_model,
            loader=train_loader, criterion=criterion,
            optimizer=optimizer, scaler=scaler, device=device,
            topo_loss=topo_loss, residual_convs=residual_convs,
            act_store=act_store, layer_cfg=layer_cfg,
            default_layer_cfg=default_layer_cfg,
            print_freq=cfg.get("print_freq", 50),
        )
        scheduler.step()

        acc = evaluate(train_model, val_loader, device)
        acc_history.append(acc)
        ce_history.append(ce_avg)
        topo_history.append(topo_avg)
        ent_history.append(ent_avg)
        kl_history.append(kl_avg)
        sparse_history.append(sparse_avg)
        grad_ent_history.append(grad_ent_avg)
        elapsed = time.time() - t0

        print(
            f"[{label}|STL] Epoch [{epoch+1:3d}/{epochs}]  "
            f"CE={ce_avg:.4f}  Topo={topo_avg:.5f}  Ent={ent_avg:.5f}  "
            f"KL={kl_avg:.5f}  Sparse={sparse_avg:.5f}  GradH={grad_ent_avg:.4f}  Val={acc:.2f}%  t={elapsed:.1f}s"
        )

        is_best = acc > best_acc
        if is_best:
            best_acc = acc
            print(f"  *** New best STL Val = {best_acc:.2f}% ***")

        ckpt = {
            "epoch": epoch, "model": base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
        }
        if is_best:
            torch.save(ckpt, ckpt_dir / f"stl_best_{label}.pt")
        if cfg.get("save_freq", 10) > 0 and (epoch + 1) % cfg["save_freq"] == 0:
            torch.save(ckpt, ckpt_dir / f"stl_epoch{epoch+1:04d}_{label}.pt")

    torch.save(ckpt, ckpt_dir / f"stl_last_{label}.pt")
    for h in hook_handles:
        h.remove()

    # Reload best weights
    best_path = ckpt_dir / f"stl_best_{label}.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        base_model.load_state_dict(best_ckpt["model"])

    loss_history = {
        "ce":           ce_history,
        "topo":         topo_history,
        "entropy":      ent_history,
        "kl":           kl_history,
        "sparse":       sparse_history,
        "grad_entropy": grad_ent_history,
    }
    return base_model, best_acc, acc_history, loss_history


def run_finetune(
    label: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    stl_val_loader: DataLoader,
    cfg: dict,
    ckpt_dir: Path,
    topo_loss: Optional[TopoLoss],
    residual_convs: dict[str, nn.Conv2d],
    layer_cfg: dict,
    default_layer_cfg: dict,
) -> tuple[float, float, list[float], list[float]]:
    """Phase 2: finetune on CIFAR-10.

    Returns (cifar_acc, stl_acc_after, cifar_acc_per_epoch, stl_acc_per_epoch).
    """
    device   = cfg["device"]
    epochs   = cfg["finetune_epochs"]
    use_cuda = device.startswith("cuda")

    base_model  = model
    train_model = model
    num_gpus    = torch.cuda.device_count() if use_cuda else 0
    if num_gpus > 1:
        train_model = nn.DataParallel(model)

    scaler = torch.amp.GradScaler("cuda") if use_cuda else None

    act_store:   dict[str, Optional[torch.Tensor]] = {n: None for n in residual_convs}
    hook_handles = []
    if topo_loss is not None:
        for name in residual_convs:
            def _hook_fn(n: str):
                def _h(_m, _i, out):
                    act_store[n] = out
                return _h
            hook_handles.append(
                residual_convs[name].register_forward_hook(_hook_fn(name))
            )

    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(base_model, cfg["finetune_lr"], cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    cifar_history:    list[float] = []
    stl_history:      list[float] = []
    ce_history:       list[float] = []
    topo_history:     list[float] = []
    ent_history:      list[float] = []
    kl_history:       list[float] = []
    sparse_history:   list[float] = []
    grad_ent_history: list[float] = []

    for epoch in range(epochs):
        t0 = time.time()
        ce_avg, topo_avg, ent_avg, kl_avg, sparse_avg, grad_ent_avg = _train_one_epoch(
            label=label, epoch=epoch, total_epochs=epochs, phase="CIFAR-FT",
            train_model=train_model, base_model=base_model,
            loader=train_loader, criterion=criterion,
            optimizer=optimizer, scaler=scaler, device=device,
            topo_loss=topo_loss, residual_convs=residual_convs,
            act_store=act_store, layer_cfg=layer_cfg,
            default_layer_cfg=default_layer_cfg,
            print_freq=cfg.get("print_freq", 50),
        )
        scheduler.step()

        cifar_acc = evaluate(train_model, val_loader,     device)
        stl_acc   = evaluate(train_model, stl_val_loader, device)
        cifar_history.append(cifar_acc)
        stl_history.append(stl_acc)
        ce_history.append(ce_avg)
        topo_history.append(topo_avg)
        ent_history.append(ent_avg)
        kl_history.append(kl_avg)
        sparse_history.append(sparse_avg)
        grad_ent_history.append(grad_ent_avg)

        print(
            f"[{label}|CIFAR-FT] Epoch [{epoch+1:3d}/{epochs}]  "
            f"CE={ce_avg:.4f}  Topo={topo_avg:.5f}  Ent={ent_avg:.5f}  "
            f"KL={kl_avg:.5f}  Sparse={sparse_avg:.5f}  GradH={grad_ent_avg:.4f}  "
            f"CIFAR={cifar_acc:.2f}%  STL={stl_acc:.2f}%  t={time.time()-t0:.1f}s"
        )

    ckpt = {
        "epoch": epochs - 1, "model": base_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(ckpt, ckpt_dir / f"finetune_last_{label}.pt")
    for h in hook_handles:
        h.remove()

    ft_loss_history = {
        "ce":           ce_history,
        "topo":         topo_history,
        "entropy":      ent_history,
        "kl":           kl_history,
        "sparse":       sparse_history,
        "grad_entropy": grad_ent_history,
    }
    return cifar_history[-1], stl_history[-1], cifar_history, stl_history, ft_loss_history


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def _sanitize_for_json(obj):
    """Recursively replace float NaN/Inf with None so json.dump produces valid JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def save_results(results: dict, out_dir: Path, label: str) -> Path:
    """Save *results* dict as JSON under *out_dir*/<label>_results.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = _sanitize_for_json(results)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{label}_results_{ts}.json"
    with open(path, "w") as fh:
        json.dump(clean, fh, indent=2)
    # Also write a canonical latest file for the analysis script
    latest = out_dir / f"{label}_results_latest.json"
    with open(latest, "w") as fh:
        json.dump(clean, fh, indent=2)
    print(f"  Results saved → {latest}")
    return latest


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict = {
    "output_dir":  None,
    "data_dir":    None,

    "num_classes": 10,
    "img_size":    96,

    # Phase 1 — STL-10
    "stl_epochs":   100,
    "lr":            1e-3,
    "weight_decay":  1e-4,

    # Phase 2 — CIFAR-10 finetune
    "finetune_epochs": 30,
    "finetune_lr":      1e-4,

    "batch_size":  128,
    "device":      "cuda",
    "num_workers": 4,
    "print_freq":  50,
    "save_freq":   10,

    "resume_pretrain": None,

    "include_downsample": False,

    "default_layer_cfg": {
        "topo_scale":     10.0,
        "factor_h":        4.0,
        "factor_w":        4.0,
        "lambda_entropy":  1.0,
        "temperature":     1.0,
    },
    "layers": {},
}


def load_config(config_path: Optional[str], extra_args: Optional[dict] = None) -> dict:
    cfg = copy.deepcopy(_DEFAULT_CONFIG)

    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            with open(p) as fh:
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
            print(f"Config loaded from: {p}")
        else:
            print(f"Config file not found ({p}), using defaults.")

    if extra_args:
        for key, val in extra_args.items():
            if val is not None:
                cfg[key] = val

    if cfg["output_dir"] is None:
        cfg["output_dir"] = str(OUTPUT_DIR)
    if cfg["data_dir"] is None:
        cfg["data_dir"] = str(BASE_DIR / "data" / "stl_cifar")

    return cfg
