"""Fashion-MNIST catastrophic forgetting experiment.

Phase 1  (pretraining):   All 10 FashionMNIST categories, full training set.
Phase 2  (finetuning):     A small dataset all labeled as one target class,
                           either pure Gaussian noise (maximum distribution
                           shift) or real images from a chosen source class
                           (class-substitution probe).

Four variants compared:
  baseline         — cross-entropy only, no regularisation
  topo_only        — TopoLoss on fc1, no KL / entropy sparsity
  topo_sparsity    — TopoLoss + KL + per-sample entropy on fc1
  topo_auxk_pooled — TopoLoss + AuxK CE loss on the downsampled cortical sheet;
                     TopK / dead-tracking on pool_h×pool_w pooled units;
                     main fc1→fc2 path uses full ReLU activations

Metrics tracked per epoch in both phases:
  ce, topo, kl, entropy, grad_entropy, val_acc (FashionMNIST val, all classes)

Additional per-class snapshots before and after finetuning:
  val_acc_per_class_before, val_acc_per_class_after

Results saved as:
  outputs/fashion_mnist_forgetting/results/fmnist_forgetting_results_latest.json
  outputs/fashion_mnist_forgetting/results/fmnist_forgetting_results_{timestamp}.json

JSON top-level keys are variant labels ("baseline", "topo_only", "topo_sparsity", "topo_auxk_pooled").
"""

# -- Imports -------------------------------------------------------------------

import argparse
import copy
import json
import math
from datetime import datetime
from pathlib import Path
import sys

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
from torch.utils.data import DataLoader, TensorDataset

from topoloss import LaplacianPyramid, TopoLoss
from topoloss.core import find_cortical_sheet_size

# -- Constants -----------------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs" / "fashion_mnist_forgetting"

FMNIST_CLASSES = [
    "T-shirt",  "Trouser", "Pullover", "Dress",  "Coat",
    "Sandal",   "Shirt",   "Sneaker",  "Bag",    "AnkleBoot",
]

VARIANT_LABELS  = ["baseline", "topo_only", "topo_sparsity", "topo_auxk_pooled"]

DISPLAY_NAMES = {
    "baseline":          "Baseline",
    "topo_only":         "Topo Only",
    "topo_sparsity":     "Topo + Sparsity",
    "topo_auxk_pooled":  "Topo + AuxK (Pooled)",
}

COLORS = {
    "baseline":          "#757575",   # gray
    "topo_only":         "#2196f3",   # blue
    "topo_sparsity":     "#4caf50",   # green
    "topo_auxk_pooled":  "#9c27b0",   # purple
}

MARKERS = {
    "baseline":          "o",
    "topo_only":         "s",
    "topo_sparsity":     "^",
    "topo_auxk_pooled":  "v",
}

# The single fc1 layer that receives TopoLoss + sparsity penalties
TOPO_LAYER_NAMES = ["fc1"]


# -- Model ---------------------------------------------------------------------

def _topk_act(pre: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the *k* largest (positive) pre-activations per row."""
    _, idx = pre.topk(k, dim=-1)
    mask = torch.zeros_like(pre, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return F.relu(pre) * mask.float()


class SimpleNN(nn.Module):
    """Two-layer MLP with centralised support for all four forgetting variants.

    sparsity_mode controls fc1 activation and AuxK dead-latent gating:
      'relu'        — standard ReLU; baseline / topo_only / topo_sparsity
      'topk_pooled' — ReLU on main fc1→fc2 path; TopK / dead-tracking on the
                      spatially-pooled cortical sheet; topo_auxk_pooled

    For the AuxK mode, auxk_loss(labels, criterion, dead_threshold) trains
    dead latents to classify via CE — no separate decoder needed.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        bias: bool = False,
        sparsity_mode: str = "relu",
        k: int = 32,
        k_aux: int = 64,
        factor_h: float = 4.0,
        factor_w: float = 4.0,
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.sparsity_mode = sparsity_mode
        self.k     = k
        self.k_aux = k_aux
        self.fc1 = nn.Linear(28 * 28, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, 10,      bias=bias)

        if sparsity_mode == "topk_pooled":
            size = find_cortical_sheet_size(hidden_size)
            self.sheet_h   = size.height
            self.sheet_w   = size.width
            self.pool_h    = max(1, round(size.height / factor_h))
            self.pool_w    = max(1, round(size.width  / factor_w))
            self.pool_size = self.pool_h * self.pool_w
            self.factor_h  = factor_h
            self.factor_w  = factor_w
            self.register_buffer("latent_counts", torch.zeros(self.pool_size))

    def reset_dead_counts(self):
        if hasattr(self, "latent_counts"):
            self.latent_counts.zero_()

    def forward(self, x):
        x_flat = x.view(-1, 28 * 28)
        pre    = self.fc1(x_flat)

        if self.sparsity_mode == "topk_pooled":
            acts = F.relu(pre)
            pool = F.adaptive_avg_pool2d(
                acts.reshape(-1, 1, self.sheet_h, self.sheet_w),
                (self.pool_h, self.pool_w),
            ).reshape(-1, self.pool_size)
            self._pre = pre
            with torch.no_grad():
                self.latent_counts += (_topk_act(pool, self.k) > 0).float().sum(0)

        else:
            acts = F.relu(pre)

        return self.fc2(acts)

    def auxk_loss(
        self,
        labels: torch.Tensor,
        criterion: nn.Module,
        dead_threshold: int = 100,
    ) -> tuple:
        """Classification AuxK loss — call immediately after forward().

        Dead pooled regions (rarely active) are upsampled to fc1 unit space;
        the top-k_aux of those fc1 units attempt to classify the batch via CE.
        Gradients flow through both fc1 (reviving dead units) and fc2.

        Returns (L_aux, dead_fraction).
        """
        dead_mask = self.latent_counts < dead_threshold
        dead_frac = dead_mask.float().mean().item()

        if not dead_mask.any():
            return self._pre.new_tensor(0.0), dead_frac

        # Upsample pool-level dead mask to fc1 unit space
        dead_spatial  = dead_mask.float().reshape(1, 1, self.pool_h, self.pool_w)
        dead_fc1_mask = F.interpolate(
            dead_spatial, (self.sheet_h, self.sheet_w), mode="nearest",
        ).reshape(self.hidden_size).bool()
        dead_pre = self._pre.clone()
        dead_pre[:, ~dead_fc1_mask] = -torch.inf
        k_eff = min(self.k_aux, int(dead_fc1_mask.sum().item()))

        if k_eff == 0:
            return self._pre.new_tensor(0.0), dead_frac

        aux_acts = _topk_act(dead_pre, k_eff)
        L_aux = criterion(self.fc2(aux_acts), labels)
        return L_aux, dead_frac


# -- Losses & metrics ----------------------------------------------------------

def cortical_sparsity_losses(
    activations: torch.Tensor,
    factor_h: float,
    factor_w: float,
    temperature: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KL-from-uniform + per-sample entropy on the downsampled cortical sheet."""
    if activations.ndim == 4:
        activations = activations.mean(dim=(2, 3))

    B, N   = activations.shape
    size   = find_cortical_sheet_size(N)
    H, W   = size.height, size.width
    sheet  = activations.reshape(B, 1, H, W)
    H_d    = max(1, round(H / factor_h))
    W_d    = max(1, round(W / factor_w))
    flat   = F.adaptive_avg_pool2d(sheet, (H_d, W_d)).reshape(B, -1)
    M      = flat.shape[1]

    probs      = F.softmax(flat / temperature, dim=-1)
    batch_mean = probs.mean(dim=0)
    kl_loss     = F.kl_div(
        torch.full_like(batch_mean, -math.log(M)),
        batch_mean, reduction="sum",
    )
    entropy_loss = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
    return kl_loss, entropy_loss


def _grad_entropy(model: nn.Module, params=None) -> float:
    """Mean normalised Shannon entropy of gradient magnitudes.

    Interpretation: 1.0 = perfectly uniform gradient; 0.0 = single-weight spike.
    Optionally pass an explicit list of tensors via `params` to restrict to
    specific layers (e.g. the cortical layer only).
    """
    entropies: list[float] = []
    for p in (params if params is not None else model.parameters()):
        if p.grad is None:
            continue
        g = p.grad.detach().float().abs().view(-1)
        if not torch.isfinite(g).all():
            continue
        total = g.sum()
        if total == 0 or g.numel() < 2:
            continue
        prob = g / total
        h = -(prob * (prob + 1e-10).log()).sum().item()
        entropies.append(h / math.log(g.numel()))
    return float(sum(entropies) / len(entropies)) if entropies else 0.0


@torch.no_grad()
def per_class_accuracy(
    model: SimpleNN,
    loader: DataLoader,
    device: str,
    n_classes: int = 10,
) -> dict[str, float]:
    """Per-class top-1 accuracy (%) on *loader*."""
    model.eval()
    correct = [0] * n_classes
    total   = [0] * n_classes
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        for cls in range(n_classes):
            mask = labels == cls
            correct[cls] += (preds[mask] == cls).sum().item()
            total[cls]   += mask.sum().item()
    model.train()
    return {
        str(cls): (100.0 * correct[cls] / total[cls]) if total[cls] > 0 else float("nan")
        for cls in range(n_classes)
    }


# -- Training phase ------------------------------------------------------------

def _build_topo_loss(model: SimpleNN, layer_cfg: dict) -> TopoLoss:
    return TopoLoss(losses=[
        LaplacianPyramid.from_layer(
            model=model, layer=model.fc1,
            factor_h=layer_cfg["fc1"]["factor_h"],
            factor_w=layer_cfg["fc1"]["factor_w"],
            scale=layer_cfg["fc1"]["topo_scale"],
        )
    ])


def _variant_config(
    model: SimpleNN,
    variant: str,
    base_layer_cfg: dict,
) -> tuple:
    """Return (topo_loss, layer_cfg) for each variant."""
    if variant == "baseline":
        return None, None

    # topo_only / topo_auxk_pooled: zero out KL and entropy lambdas
    layer_cfg_copy = copy.deepcopy(base_layer_cfg)
    if variant in ("topo_only", "topo_auxk_pooled"):
        for lc in layer_cfg_copy.values():
            lc["lambda_kl"]      = 0.0
            lc["lambda_entropy"] = 0.0

    return _build_topo_loss(model, base_layer_cfg), layer_cfg_copy


def run_phase(
    label: str,
    phase: str,          # "pretrain" or "finetune" (used in ckpt names + prints)
    model: SimpleNN,
    train_loader: DataLoader,
    monitor_loader: DataLoader,  # always FMNIST val — tracks forgetting
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int,
    device: str,
    print_freq: int,
    topo_loss=None,
    layer_cfg: dict = None,
    ckpt_dir: Path = None,
) -> dict:
    """Run one training phase; return per-epoch loss/metric histories.

    Returns
    -------
    dict with keys:
        ce, topo, kl, entropy, grad_entropy, val_acc
    Each value is a list[float] of length *epochs*.
    """
    history = {k: [] for k in (
        "ce", "topo", "kl", "entropy", "grad_entropy", "val_acc",
        "auxk_aux", "auxk_dead_frac",
    )}

    # Activation hooks for KL/entropy penalties
    act_store: dict = {n: None for n in TOPO_LAYER_NAMES}
    hook_handles: list = []
    if topo_loss is not None:
        for name in TOPO_LAYER_NAMES:
            layer = getattr(model, name)
            def _make_hook(n):
                def _h(_m, _i, out):
                    act_store[n] = out
                return _h
            hook_handles.append(layer.register_forward_hook(_make_hook(name)))

    best_acc = 0.0
    is_auxk  = model.sparsity_mode != "relu"
    if is_auxk:
        model.reset_dead_counts()

    for epoch in range(epochs):
        model.train()
        sum_ce = sum_topo = 0.0
        sum_kl  = {n: 0.0 for n in TOPO_LAYER_NAMES}
        sum_ent = {n: 0.0 for n in TOPO_LAYER_NAMES}
        sum_grad_ent = 0.0
        sum_auxk_aux = 0.0
        sum_auxk_dead  = 0.0
        n_total = n_steps = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(imgs)
            ce     = criterion(logits, labels)

            extra = torch.zeros(1, device=device)
            if topo_loss is not None:
                topo     = topo_loss.compute(model=model, reduce_mean=True)
                sum_topo += topo.item() * imgs.size(0)
                extra    = extra + topo

                if is_auxk:
                    lc = layer_cfg[TOPO_LAYER_NAMES[0]]
                    L_aux, d_frac = model.auxk_loss(
                        labels, criterion,
                        int(lc.get("dead_threshold", 100)),
                    )
                    alpha = float(lc.get("auxk_alpha", 1 / 32))
                    extra = extra + alpha * L_aux
                    sum_auxk_aux   += L_aux.item() * imgs.size(0)
                    sum_auxk_dead  += d_frac
                else:
                    for name in TOPO_LAYER_NAMES:
                        act = act_store[name]
                        if act is not None:
                            lc = layer_cfg[name]
                            kl, ent = cortical_sparsity_losses(
                                act, lc["factor_h"], lc["factor_w"],
                                lc.get("temperature", 3.0),
                            )
                            extra    = extra + lc["lambda_kl"] * kl + lc["lambda_entropy"] * ent
                            sum_kl[name]  += kl.item()  * imgs.size(0)
                            sum_ent[name] += ent.item() * imgs.size(0)

            (ce + extra).backward()
            sum_grad_ent += _grad_entropy(model)
            optimizer.step()

            bs       = imgs.size(0)
            sum_ce  += ce.item() * bs
            n_total += bs
            n_steps += 1

        # Evaluate on FashionMNIST val (always!)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in monitor_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (model(imgs).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        model.train()
        val_acc = 100.0 * correct / total

        ce_avg  = sum_ce   / n_total
        topo_avg = sum_topo / n_total
        kl_avg   = sum(sum_kl[n]  for n in TOPO_LAYER_NAMES) / n_total
        ent_avg  = sum(sum_ent[n] for n in TOPO_LAYER_NAMES) / n_total
        ge_avg   = sum_grad_ent / max(n_steps, 1)

        history["ce"].append(ce_avg)
        history["topo"].append(topo_avg)
        history["kl"].append(kl_avg)
        history["entropy"].append(ent_avg)
        history["grad_entropy"].append(ge_avg)
        history["val_acc"].append(val_acc)

        auxk_aux_avg   = sum_auxk_aux   / n_total if n_total else 0.0
        auxk_dead_avg  = sum_auxk_dead  / max(n_steps, 1)
        history["auxk_aux"].append(auxk_aux_avg)
        history["auxk_dead_frac"].append(auxk_dead_avg)

        if ckpt_dir is not None and (phase == "pretrain") and val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "epoch":         epoch,
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "best_acc":      best_acc,
                "sparsity_mode": model.sparsity_mode,
                "k":             model.k,
                "k_aux":         model.k_aux,
                "hidden_size":   model.hidden_size,
                "factor_h":      getattr(model, "factor_h", None),
                "factor_w":      getattr(model, "factor_w", None),
            }, ckpt_dir / f"best_{phase}_{label}.pt")

        if (epoch + 1) % print_freq == 0 or epoch == epochs - 1:
            if is_auxk:
                print(
                    f"  [{label}|{phase}] Epoch [{epoch+1:3d}/{epochs}]  "
                    f"CE={ce_avg:.4f}  Topo={topo_avg:.6f}  "
                    f"AuxK={auxk_aux_avg:.4f}  "
                    f"Dead={auxk_dead_avg:.1%}  "
                    f"GradH={ge_avg:.4f}  Val={val_acc:.1f}%"
                )
            elif topo_loss is not None:
                print(
                    f"  [{label}|{phase}] Epoch [{epoch+1:3d}/{epochs}]  "
                    f"CE={ce_avg:.4f}  Topo={topo_avg:.6f}  "
                    f"KL={kl_avg:.4f}  Ent={ent_avg:.4f}  "
                    f"GradH={ge_avg:.4f}  Val={val_acc:.1f}%"
                )
            else:
                print(
                    f"  [{label}|{phase}] Epoch [{epoch+1:3d}/{epochs}]  "
                    f"CE={ce_avg:.4f}  GradH={ge_avg:.4f}  Val={val_acc:.1f}%"
                )

    if ckpt_dir is not None:
        torch.save({
            "epoch":         epochs - 1,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "sparsity_mode": model.sparsity_mode,
            "k":             model.k,
            "k_aux":         model.k_aux,
            "hidden_size":   model.hidden_size,
            "factor_h":      getattr(model, "factor_h", None),
            "factor_w":      getattr(model, "factor_w", None),
        }, ckpt_dir / f"last_{phase}_{label}.pt")

    for h in hook_handles:
        h.remove()

    return history


# -- Experiment entry point ----------------------------------------------------

def train(cfg: dict) -> None:
    print("=" * 70)
    print("  FASHION-MNIST CATASTROPHIC FORGETTING EXPERIMENT")
    print("=" * 70)
    print(json.dumps({k: v for k, v in cfg.items() if k != "layers"}, indent=2))
    print("  layers:")
    for lname, lvals in cfg["layers"].items():
        print(f"    {lname}: {json.dumps(lvals)}")
    print()

    # Device
    if torch.cuda.is_available() and cfg["device"].startswith("cuda"):
        device = cfg["device"]
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}\n")

    out_dir  = Path(cfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    res_dir  = out_dir / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # FashionMNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_ds = datasets.FashionMNIST(cfg["data_dir"], train=True,  download=True, transform=transform)
    val_ds   = datasets.FashionMNIST(cfg["data_dir"], train=False, download=True, transform=transform)
    _pin = device.startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=2, pin_memory=_pin)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=2, pin_memory=_pin)
    print(f"FashionMNIST: {len(train_ds):,} train  |  {len(val_ds):,} val  |  10 classes\n")

    # Finetune dataset
    ft_target = int(cfg.get("noise_target_class", 0))
    ft_source = cfg.get("ft_source_class", None)
    n_ft      = int(cfg.get("noise_samples", 2000))
    if ft_source is None:
        # Noise mode: pure Gaussian noise labeled as ft_target
        ft_imgs     = torch.randn(n_ft, 1, 28, 28)
        ft_mode_str = "noise"
    else:
        # Class-substitution mode: real images from ft_source, relabeled as ft_target
        ft_source   = int(ft_source)
        src_indices = [i for i, (_, lbl) in enumerate(train_ds) if int(lbl) == ft_source]
        rng         = torch.Generator().manual_seed(42)
        chosen      = torch.randint(len(src_indices), (n_ft,), generator=rng).tolist()
        ft_imgs     = torch.stack([train_ds[src_indices[i]][0] for i in chosen])
        ft_mode_str = f"{ft_source} ({FMNIST_CLASSES[ft_source]})"
    ft_labels  = torch.full((n_ft,), ft_target, dtype=torch.long)
    ft_ds      = TensorDataset(ft_imgs, ft_labels)
    ft_loader  = DataLoader(ft_ds, batch_size=cfg["batch_size"], shuffle=True)
    print(
        f"Finetune: target={ft_target} ({FMNIST_CLASSES[ft_target]})  "
        f"source={ft_mode_str}  |  {n_ft} samples"
    )
    print()

    layer_cfg = cfg["layers"]
    criterion = nn.CrossEntropyLoss().to(device)

    all_results: dict = {}

    for variant in VARIANT_LABELS:
        print("\n" + "=" * 70)
        print(f"  VARIANT: {DISPLAY_NAMES[variant]}")
        print("=" * 70)

        # ── Phase 1: Pretrain on FashionMNIST ──────────────────────────────
        print(f"\n  Phase 1: Pretraining on all 10 FashionMNIST classes ({cfg['pretrain_epochs']} epochs)")
        hs = cfg.get("hidden_size", 256)
        if variant == "topo_auxk_pooled":
            lc = layer_cfg["fc1"]
            model = SimpleNN(
                hidden_size=hs,
                sparsity_mode="topk_pooled",
                factor_h=lc.get("factor_h", 4.0),
                factor_w=lc.get("factor_w", 4.0),
                k=cfg.get("auxk_k_pooled", 4),
                k_aux=cfg.get("auxk_k_aux_pooled", 8),
            ).to(device)
        else:
            model = SimpleNN(hidden_size=hs).to(device)
        optimizer  = optim.Adam(model.parameters(), lr=cfg["lr"])
        topo_loss, eff_layer_cfg = _variant_config(model, variant, layer_cfg)

        pretrain_history = run_phase(
            label=variant, phase="pretrain",
            model=model, train_loader=train_loader, monitor_loader=val_loader,
            criterion=criterion, optimizer=optimizer,
            epochs=cfg["pretrain_epochs"], device=device,
            print_freq=cfg["print_freq"],
            topo_loss=topo_loss, layer_cfg=eff_layer_cfg,
            ckpt_dir=ckpt_dir,
        )

        val_acc_before      = pretrain_history["val_acc"][-1]
        cls_acc_before      = per_class_accuracy(model, val_loader, device)
        print(f"\n  Pre-finetune val acc: {val_acc_before:.1f}%")

        # ── Phase 2: Finetune ──────────────────────────────────────────────
        print(
            f"\n  Phase 2: Finetuning — target={ft_target} ({FMNIST_CLASSES[ft_target]})  "
            f"source={ft_mode_str}  ({cfg['finetune_epochs']} epochs)"
        )
        ft_optimizer = optim.Adam(model.parameters(), lr=cfg["finetune_lr"])
        ft_topo_loss, ft_layer_cfg = _variant_config(model, variant, layer_cfg)

        ft_history = run_phase(
            label=variant, phase="finetune",
            model=model, train_loader=ft_loader, monitor_loader=val_loader,
            criterion=criterion, optimizer=ft_optimizer,
            epochs=cfg["finetune_epochs"], device=device,
            print_freq=cfg["print_freq"],
            topo_loss=ft_topo_loss, layer_cfg=ft_layer_cfg,
            ckpt_dir=ckpt_dir,
        )

        val_acc_after  = ft_history["val_acc"][-1]
        cls_acc_after  = per_class_accuracy(model, val_loader, device)
        forgetting     = val_acc_before - val_acc_after
        print(f"\n  Post-finetune val acc: {val_acc_after:.1f}%  (forgetting: {forgetting:+.1f}pp)")

        all_results[variant] = {
            # Accuracy trajectories
            "pretrain_val_acc_per_epoch":      pretrain_history["val_acc"],
            "ft_val_acc_per_epoch":            ft_history["val_acc"],
            # Before / after snapshots
            "val_acc_before":                  val_acc_before,
            "val_acc_after":                   val_acc_after,
            "forgetting_pp":                   forgetting,
            "val_acc_per_class_before":        cls_acc_before,
            "val_acc_per_class_after":         cls_acc_after,
            # Loss curves — pretrain
            "pretrain_ce_per_epoch":           pretrain_history["ce"],
            "pretrain_topo_per_epoch":         pretrain_history["topo"],
            "pretrain_kl_per_epoch":           pretrain_history["kl"],
            "pretrain_entropy_per_epoch":      pretrain_history["entropy"],
            "pretrain_grad_entropy_per_epoch": pretrain_history["grad_entropy"],
            "pretrain_auxk_aux_per_epoch":     pretrain_history["auxk_aux"],
            "pretrain_auxk_dead_frac_per_epoch": pretrain_history["auxk_dead_frac"],
            # Loss curves — finetune
            "ft_ce_per_epoch":                 ft_history["ce"],
            "ft_topo_per_epoch":               ft_history["topo"],
            "ft_kl_per_epoch":                 ft_history["kl"],
            "ft_entropy_per_epoch":            ft_history["entropy"],
            "ft_grad_entropy_per_epoch":       ft_history["grad_entropy"],
            "ft_auxk_aux_per_epoch":           ft_history["auxk_aux"],
            "ft_auxk_dead_frac_per_epoch":     ft_history["auxk_dead_frac"],
            # Metadata
            "ft_target_class":                 ft_target,
            "ft_target_name":                  FMNIST_CLASSES[ft_target],
            "ft_source_class":                 ft_source,
            "ft_source_name":                  FMNIST_CLASSES[ft_source] if ft_source is not None else None,
            "ft_mode":                         ft_mode_str,
            "config":                          {k: v for k, v in cfg.items() if k != "layers"},
        }

    # ── Save results ──────────────────────────────────────────────────────────
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_path   = res_dir / f"fmnist_forgetting_results_{ts}.json"
    last_path = res_dir / "fmnist_forgetting_results_latest.json"

    for path in (ts_path, last_path):
        with open(path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {last_path}")
    print("  Run analyze_fmnist_forgetting.py to generate figures.")
    print("\nForgetting summary:")
    for v, r in all_results.items():
        print(f"  {DISPLAY_NAMES[v]:22s}  before={r['val_acc_before']:.1f}%  after={r['val_acc_after']:.1f}%  Δ={r['forgetting_pp']:+.1f}pp")


# -- Config & entry point ------------------------------------------------------

DEFAULT_CFG = {
    "data_dir":           None,
    "output_dir":         None,
    "hidden_size":        256,
    "pretrain_epochs":    20,
    "finetune_epochs":    30,
    "batch_size":         128,
    "lr":                 5e-4,
    "finetune_lr":        5e-4,
    "device":             "cuda:0",
    "print_freq":         5,
    "noise_target_class": 7,
    "noise_samples":      2000,
    "ft_source_class":    None,
    "auxk_k":             32,
    "auxk_k_aux":         64,
    "auxk_k_pooled":      4,
    "auxk_k_aux_pooled":  8,
    "layers": {
        "fc1": {
            "topo_scale":    10.0,
            "factor_h":       4.0,
            "factor_w":       4.0,
            "lambda_kl":        0.0,
            "lambda_entropy":   5.0,
            "temperature":      1.0,
            "auxk_alpha":       0.03125,
            "dead_threshold":   100,
        }
    },
}


def _load_json(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def get_config() -> dict:
    p = argparse.ArgumentParser(
        description="FashionMNIST catastrophic forgetting experiment"
    )
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir",           default=None)
    p.add_argument("--output-dir",         default=None)
    p.add_argument("--pretrain-epochs",    type=int,   default=None)
    p.add_argument("--finetune-epochs",    type=int,   default=None)
    p.add_argument("--batch-size",         type=int,   default=None)
    p.add_argument("--lr",                 type=float, default=None)
    p.add_argument("--finetune-lr",        type=float, default=None)
    p.add_argument("--device",             default=None)
    p.add_argument("--print-freq",         type=int,   default=None)
    p.add_argument("--noise-target-class", type=int,   default=None)
    p.add_argument("--noise-samples",      type=int,   default=None)
    p.add_argument("--ft-source-class",    type=int,   default=None,
                   help="Source class whose images are used as finetune stimuli "
                        "(relabeled as noise-target-class). Omit for pure-noise mode.")
    p.add_argument("--auxk-k",             type=int,   default=None)
    p.add_argument("--auxk-k-aux",         type=int,   default=None)
    p.add_argument("--auxk-k-pooled",      type=int,   default=None)
    p.add_argument("--auxk-k-aux-pooled",  type=int,   default=None)
    cli = p.parse_args()

    cfg = copy.deepcopy(DEFAULT_CFG)

    # JSON config override
    default_json = str(BASE_DIR / "configs" / "fashion_mnist_forgetting.json")
    config_path  = cli.config or (default_json if Path(default_json).exists() else None)
    if config_path:
        file_cfg = _load_json(config_path)
        for k, v in file_cfg.items():
            if k == "layers":
                for lname, lvals in v.items():
                    cfg["layers"].setdefault(lname, {}).update(lvals)
            elif v is not None:
                cfg[k] = v

    # CLI overrides
    overrides = {
        "data_dir":           cli.data_dir,
        "output_dir":         cli.output_dir,
        "pretrain_epochs":    cli.pretrain_epochs,
        "finetune_epochs":    cli.finetune_epochs,
        "batch_size":         cli.batch_size,
        "lr":                 cli.lr,
        "finetune_lr":        cli.finetune_lr,
        "device":             cli.device,
        "print_freq":         cli.print_freq,
        "noise_target_class": cli.noise_target_class,
        "noise_samples":      cli.noise_samples,
        "ft_source_class":    cli.ft_source_class,
        "auxk_k":             cli.auxk_k,
        "auxk_k_aux":         cli.auxk_k_aux,
        "auxk_k_pooled":      cli.auxk_k_pooled,
        "auxk_k_aux_pooled":  cli.auxk_k_aux_pooled,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    # Resolve null paths
    if not cfg["data_dir"]:
        cfg["data_dir"] = str(BASE_DIR / "data")
    if not cfg["output_dir"]:
        cfg["output_dir"] = str(OUTPUT_DIR)

    return cfg


if __name__ == "__main__":
    cfg = get_config()
    train(cfg)
