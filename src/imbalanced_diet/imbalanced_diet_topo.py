"""imbalanced_diet_topo.py

Experiment studying how imbalanced training data shapes topographic cortical
organisation in two-layer MLPs trained on a 4-class FashionMNIST subset.

4 classes selected (and remapped to 0-3):
  0  T-shirt/top   ("top")
  1  Trouser       ("bottom")
  7  Sneaker       ("footwear")
  8  Bag           ("accessory")

4 model variants:
  baseline   — cross-entropy only, no regularisation
  topo_r2    — TopoLoss (τ=5, factor=2, ~8×8 pooled regions) + KL/entropy sparsity
  topo_r4    — TopoLoss (τ=5, factor=4, ~4×4 pooled regions) + KL/entropy sparsity
  topo_r8    — TopoLoss (τ=5, factor=8, ~2×2 pooled regions) + KL/entropy sparsity

6 training diets (T-shirt / Trouser / Sneaker / Bag class proportions):
  balanced            25 / 25 / 25 / 25
  top_dominant        70 / 10 / 10 / 10
  bottom_dominant     10 / 70 / 10 / 10
  footwear_dominant   10 / 10 / 70 / 10
  bag_dominant        10 / 10 / 10 / 70
  extreme_skew        55 / 25 / 15 /  5

All 24 (diet × variant) combinations are trained. Results, per-neuron
selectivity t-statistics, and spatial-clustering scores are written to:
  outputs/imbalanced_diet/results/imbalanced_diet_results_latest.json

Run visualisation with:
  python src/imbalanced_diet/analyze_imbalanced_diet.py
"""

from __future__ import annotations

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

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs" / "imbalanced_diet"

# The 4 original FashionMNIST class indices we keep
ORIG_CLASS_INDICES = [0, 1, 7, 8]
# Human-readable names in new 0-3 label space
CLASS_NAMES = ["T-shirt", "Trouser", "Sneaker", "Bag"]
N_CLASSES   = 4
# Mapping: original FMNIST label → new 0-3 label
ORIG_TO_NEW: dict[int, int] = {o: n for n, o in enumerate(ORIG_CLASS_INDICES)}

# Per-class colours used in every figure (RGB 0-255)
CLASS_RGB = {
    "T-shirt": (213,  62,  79),   # crimson
    "Trouser": ( 50, 100, 193),   # royal blue
    "Sneaker": ( 44, 160,  44),   # forest green
    "Bag":     (230, 115,   0),   # dark orange
}
CLASS_HEX = {
    "T-shirt": "#D53E4F",
    "Trouser": "#3264C1",
    "Sneaker": "#2CA02C",
    "Bag":     "#E67300",
}

# Training diets: [T-shirt, Trouser, Sneaker, Bag] proportions
DIETS: dict[str, list[float]] = {
    "balanced":          [0.25, 0.25, 0.25, 0.25],
    "top_dominant":      [0.70, 0.10, 0.10, 0.10],
    "bottom_dominant":   [0.10, 0.70, 0.10, 0.10],
    "footwear_dominant": [0.10, 0.10, 0.70, 0.10],
    "bag_dominant":      [0.10, 0.10, 0.10, 0.70],
    "extreme_skew":      [0.55, 0.25, 0.15, 0.05],
}

DIET_LABELS: dict[str, str] = {
    "balanced":          "Balanced\n25/25/25/25",
    "top_dominant":      "Top Dom.\n70/10/10/10",
    "bottom_dominant":   "Bottom Dom.\n10/70/10/10",
    "footwear_dominant": "Footwear Dom.\n10/10/70/10",
    "bag_dominant":      "Bag Dom.\n10/10/10/70",
    "extreme_skew":      "Extreme Skew\n55/25/15/5",
}

# Model variants: no topo for baseline; factor controls pyramid downsampling
VARIANTS: dict[str, dict] = {
    "baseline": {"topo_scale": 0.0, "factor_h": 4.0, "factor_w": 4.0,
                 "lambda_kl": 0.0, "lambda_entropy": 0.0},
    "topo_r2":  {"topo_scale": 5.0, "factor_h": 2.0, "factor_w": 2.0,
                 "lambda_kl": 0.1, "lambda_entropy": 2.0},
    "topo_r4":  {"topo_scale": 5.0, "factor_h": 4.0, "factor_w": 4.0,
                 "lambda_kl": 0.1, "lambda_entropy": 2.0},
    "topo_r8":  {"topo_scale": 5.0, "factor_h": 8.0, "factor_w": 8.0,
                 "lambda_kl": 0.1, "lambda_entropy": 2.0},
}

VARIANT_LABELS: dict[str, str] = {
    "baseline": "Baseline (CE only)",
    "topo_r2":  "Topo τ=5  r=2  (8×8 regions)",
    "topo_r4":  "Topo τ=5  r=4  (4×4 regions)",
    "topo_r8":  "Topo τ=5  r=8  (2×2 regions)",
}

TOPO_LAYER_NAMES = ["fc1"]


# ── Model ─────────────────────────────────────────────────────────────────────

class SimpleNN4(nn.Module):
    """Two-layer MLP for 4-class FashionMNIST (same fc1 architecture as the
    existing experiments so the same TopoLoss applies to fc1).

    Attributes:
        _fc1_acts  (tensor, set in forward): post-ReLU fc1 activations for the
                   current batch; used by the sparsity losses and analysis.
    """

    def __init__(self, hidden_size: int = 256, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(28 * 28, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, N_CLASSES, bias=bias)
        self._fc1_acts: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28 * 28)
        acts = F.relu(self.fc1(x))
        self._fc1_acts = acts.detach()
        return self.fc2(acts)


# ── Dataset utilities ─────────────────────────────────────────────────────────

def _extract_4class_splits(
    full_ds: torch.utils.data.Dataset,
) -> dict[int, list[torch.Tensor]]:
    """Return dict new_label → list of image tensors (one per class)."""
    per_class: dict[int, list[torch.Tensor]] = {c: [] for c in range(N_CLASSES)}
    targets = full_ds.targets if isinstance(full_ds.targets, list) \
              else full_ds.targets.tolist()
    for i, lbl in enumerate(targets):
        lbl = int(lbl)
        if lbl in ORIG_TO_NEW:
            per_class[ORIG_TO_NEW[lbl]].append(full_ds[i][0])
    return per_class


def build_diet_dataset(
    per_class_imgs: dict[int, list[torch.Tensor]],
    proportions: list[float],
    total_samples: int = 10_000,
    seed: int = 42,
) -> TensorDataset:
    """Sample per-class images according to *proportions* summing to 1."""
    rng = np.random.default_rng(seed)
    imgs_list:   list[torch.Tensor] = []
    labels_list: list[int]          = []

    for cls_idx, prop in enumerate(proportions):
        n_want = max(1, round(total_samples * prop))
        pool   = per_class_imgs[cls_idx]
        n_pool = len(pool)
        if n_want <= n_pool:
            chosen_idx = rng.choice(n_pool, size=n_want, replace=False).tolist()
        else:
            chosen_idx = rng.choice(n_pool, size=n_want, replace=True).tolist()
        imgs_list.extend(pool[i] for i in chosen_idx)
        labels_list.extend([cls_idx] * n_want)

    imgs_t   = torch.stack(imgs_list)
    labels_t = torch.tensor(labels_list, dtype=torch.long)
    perm     = torch.randperm(len(imgs_t),
                              generator=torch.Generator().manual_seed(seed))
    return TensorDataset(imgs_t[perm], labels_t[perm])


def build_balanced_val_dataset(
    per_class_imgs: dict[int, list[torch.Tensor]],
) -> TensorDataset:
    """All available validation images for the 4 classes (unmodified split)."""
    imgs_list:   list[torch.Tensor] = []
    labels_list: list[int]          = []
    for cls_idx in range(N_CLASSES):
        imgs_list.extend(per_class_imgs[cls_idx])
        labels_list.extend([cls_idx] * len(per_class_imgs[cls_idx]))
    return TensorDataset(torch.stack(imgs_list),
                         torch.tensor(labels_list, dtype=torch.long))


# ── Loss helpers ──────────────────────────────────────────────────────────────

def _build_topo_loss(
    model: SimpleNN4,
    topo_scale: float,
    factor_h: float,
    factor_w: float,
) -> TopoLoss | None:
    if topo_scale <= 0.0:
        return None
    return TopoLoss(losses=[
        LaplacianPyramid.from_layer(
            model=model,
            layer=model.fc1,
            factor_h=factor_h,
            factor_w=factor_w,
            scale=topo_scale,
        )
    ])


def cortical_sparsity_losses(
    activations: torch.Tensor,
    factor_h: float,
    factor_w: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KL-from-uniform + per-sample entropy on the downsampled cortical sheet.

    Uses non-negative pooled activations with L1 normalisation so that true
    ReLU zeros count as zero probability mass (rewards genuine sparsity).
    """
    B, N   = activations.shape
    size   = find_cortical_sheet_size(N)
    H, W   = size.height, size.width
    sheet  = activations.reshape(B, 1, H, W)
    H_d    = max(1, round(H / factor_h))
    W_d    = max(1, round(W / factor_w))
    flat   = F.adaptive_avg_pool2d(sheet, (H_d, W_d)).reshape(B, -1)
    M      = flat.shape[1]

    mags    = flat.clamp(min=0.0)
    totals  = mags.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(mags, 1.0 / M)
    probs   = torch.where(totals > 0,
                          mags / totals.clamp(min=1e-10),
                          uniform)
    batch_mean = probs.mean(dim=0)
    kl_loss    = math.log(M) + (batch_mean * (batch_mean + 1e-10).log()).sum()
    ent_loss   = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
    return kl_loss, ent_loss


@torch.no_grad()
def _activation_entropy(acts: torch.Tensor) -> float:
    """Normalised Shannon entropy of post-ReLU fc1 activation magnitudes."""
    a = acts.detach().float()
    mags   = a.abs()
    totals = mags.sum(dim=-1, keepdim=True)
    valid  = totals.squeeze(-1) > 0
    if not valid.any():
        return 0.0
    mags   = mags[valid]
    totals = totals[valid]
    prob   = mags / totals.clamp(min=1e-10)
    h      = -(prob * (prob + 1e-10).log()).sum(dim=-1)
    return (h / math.log(a.shape[1])).mean().item()


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_run(
    model: SimpleNN4,
    train_loader: DataLoader,
    val_loader: DataLoader,
    variant_cfg: dict,
    epochs: int,
    lr: float,
    device: str,
    print_freq: int = 5,
    label: str = "",
) -> dict:
    """Train model for *epochs* and return per-epoch history dict."""
    topo_loss  = _build_topo_loss(
        model,
        variant_cfg["topo_scale"],
        variant_cfg["factor_h"],
        variant_cfg["factor_w"],
    )
    lambda_kl  = variant_cfg["lambda_kl"]
    lambda_ent = variant_cfg["lambda_entropy"]
    factor_h   = variant_cfg["factor_h"]
    factor_w   = variant_cfg["factor_w"]

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list] = {
        "ce": [], "topo": [], "kl": [], "entropy": [],
        "val_acc": [], "val_acc_per_class": [],
    }

    for epoch in range(epochs):
        model.train()
        sum_ce   = sum_topo = sum_kl = sum_ent = 0.0
        n_total  = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(imgs)
            ce     = criterion(logits, labels)
            extra  = ce.new_tensor(0.0)

            if topo_loss is not None:
                topo   = topo_loss.compute(model=model, reduce_mean=True)
                extra  = extra + topo
                sum_topo += topo.item() * imgs.size(0)

                act = model._fc1_acts
                if act is not None and (lambda_kl > 0.0 or lambda_ent > 0.0):
                    kl, ent = cortical_sparsity_losses(act, factor_h, factor_w)
                    extra   = extra + lambda_kl * kl + lambda_ent * ent
                    sum_kl  += kl.item()  * imgs.size(0)
                    sum_ent += ent.item() * imgs.size(0)

            (ce + extra).backward()
            optimizer.step()

            sum_ce  += ce.item() * imgs.size(0)
            n_total += imgs.size(0)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        correct       = 0
        total_val     = 0
        cls_correct   = [0] * N_CLASSES
        cls_total     = [0] * N_CLASSES
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct    += (preds == labels).sum().item()
                total_val  += labels.size(0)
                for c in range(N_CLASSES):
                    mask = labels == c
                    cls_correct[c] += (preds[mask] == c).sum().item()
                    cls_total[c]   += mask.sum().item()
        model.train()

        val_acc          = 100.0 * correct / total_val
        val_acc_per_cls  = [
            100.0 * cls_correct[c] / cls_total[c] if cls_total[c] > 0 else float("nan")
            for c in range(N_CLASSES)
        ]

        history["ce"].append(sum_ce   / n_total)
        history["topo"].append(sum_topo / n_total)
        history["kl"].append(sum_kl   / n_total)
        history["entropy"].append(sum_ent  / n_total)
        history["val_acc"].append(val_acc)
        history["val_acc_per_class"].append(val_acc_per_cls)

        if (epoch + 1) % print_freq == 0 or epoch == epochs - 1:
            per_cls_str = "  ".join(
                f"{CLASS_NAMES[c]}={v:.1f}%" for c, v in enumerate(val_acc_per_cls)
            )
            if topo_loss is not None:
                print(
                    f"  [{label}] E{epoch+1:3d}/{epochs}  "
                    f"CE={sum_ce/n_total:.4f}  Topo={sum_topo/n_total:.5f}  "
                    f"KL={sum_kl/n_total:.4f}  Ent={sum_ent/n_total:.4f}  "
                    f"Val={val_acc:.1f}%  |  {per_cls_str}"
                )
            else:
                print(
                    f"  [{label}] E{epoch+1:3d}/{epochs}  "
                    f"CE={sum_ce/n_total:.4f}  Val={val_acc:.1f}%  |  {per_cls_str}"
                )

    return history


# ── Post-training analysis ────────────────────────────────────────────────────

@torch.no_grad()
def compute_t_stats(
    model: SimpleNN4,
    val_loader: DataLoader,
    device: str,
) -> np.ndarray:
    """Per-neuron Welch t-statistics for each class vs. rest.

    Returns ndarray of shape (n_neurons, N_CLASSES).
    t_stats[i, c] > 0  means neuron i fires more for class c than all others.
    """
    model.eval()
    all_acts:   list[np.ndarray] = []
    all_labels: list[int]        = []

    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        model(imgs)
        acts = model._fc1_acts          # (B, hidden)
        all_acts.append(acts.cpu().numpy())
        all_labels.extend(labels.numpy().tolist())

    A = np.concatenate(all_acts, axis=0)   # (N, hidden)
    L = np.array(all_labels)               # (N,)
    n_neurons = A.shape[1]
    t_out = np.zeros((n_neurons, N_CLASSES), dtype=np.float32)

    for c in range(N_CLASSES):
        mask_c  = L == c
        mask_r  = ~mask_c
        if mask_c.sum() < 2 or mask_r.sum() < 2:
            continue
        tgt   = A[mask_c]     # (n_c, n_neurons)
        rest  = A[mask_r]     # (n_r, n_neurons)
        mu_t  = tgt.mean(0)
        mu_r  = rest.mean(0)
        var_t = tgt.var(0,  ddof=1)
        var_r = rest.var(0, ddof=1)
        n_t, n_r = len(tgt), len(rest)
        se = np.sqrt(var_t / n_t + var_r / n_r + 1e-10)
        t_out[:, c] = (mu_t - mu_r) / se

    return t_out


@torch.no_grad()
def compute_mean_activation_heatmap(
    model: SimpleNN4,
    val_loader: DataLoader,
    device: str,
) -> np.ndarray:
    """Mean post-ReLU activation per neuron (shape: n_neurons).  Normalised to [0,1]."""
    model.eval()
    accum   = np.zeros(model.hidden_size, dtype=np.float64)
    n_total = 0
    for imgs, _ in val_loader:
        imgs = imgs.to(device)
        model(imgs)
        accum   += model._fc1_acts.cpu().numpy().sum(axis=0)
        n_total += imgs.size(0)
    hm = accum / max(n_total, 1)
    mx = hm.max()
    return (hm / mx).astype(np.float32) if mx > 0 else hm.astype(np.float32)


@torch.no_grad()
def compute_class_activation_heatmaps(
    model: SimpleNN4,
    val_loader: DataLoader,
    device: str,
) -> np.ndarray:
    """Per-class mean activation heatmap.  Returns (N_CLASSES, n_neurons)."""
    model.eval()
    accum   = np.zeros((N_CLASSES, model.hidden_size), dtype=np.float64)
    counts  = np.zeros(N_CLASSES, dtype=np.int64)

    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        model(imgs)
        acts = model._fc1_acts.cpu().numpy()   # (B, hidden)
        for c in range(N_CLASSES):
            mask = (labels == c).cpu().numpy()
            if mask.any():
                accum[c]  += acts[mask].sum(axis=0)
                counts[c] += mask.sum()

    for c in range(N_CLASSES):
        if counts[c] > 0:
            accum[c] /= counts[c]

    return accum.astype(np.float32)   # (N_CLASSES, n_neurons)


def compute_spatial_clustering_score(
    t_stats: np.ndarray,
    hidden_size: int,
) -> float:
    """Spatial Contiguity Score (SCS): fraction of 4-connected neighbours that
    share the same dominant class, averaged over all neurons.

    Range: [0, 1].  0 = maximally disordered; 1 = perfectly clustered.
    Expected value under random class assignment = sum(p_c^2).
    """
    size = find_cortical_sheet_size(hidden_size)
    H, W = size.height, size.width
    t_pos     = np.maximum(t_stats, 0.0)            # (n_neurons, N_CLASSES)
    dom_class = t_pos.argmax(axis=1).reshape(H, W)  # (H, W) dominant class map

    same_neighbour_count = np.zeros((H, W), dtype=np.float32)
    num_neighbours       = np.zeros((H, W), dtype=np.float32)

    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in shifts:
        r_s = max(0,  dr);  r_e = H + min(0, dr)
        c_s = max(0,  dc);  c_e = W + min(0, dc)
        r_n = max(0, -dr);  r_ne = H + min(0, -dr)
        c_n = max(0, -dc);  c_ne = W + min(0, -dc)
        same_neighbour_count[r_s:r_e, c_s:c_e] += (
            dom_class[r_s:r_e, c_s:c_e] == dom_class[r_n:r_ne, c_n:c_ne]
        ).astype(np.float32)
        num_neighbours[r_s:r_e, c_s:c_e] += 1.0

    frac = same_neighbour_count / num_neighbours.clip(min=1)
    return float(frac.mean())


def compute_class_territories(t_stats: np.ndarray) -> list[float]:
    """Fraction of neurons whose dominant class is each class c.

    Returns list of length N_CLASSES summing to 1.0.
    """
    t_pos     = np.maximum(t_stats, 0.0)
    dom_class = t_pos.argmax(axis=1)   # (n_neurons,)
    n         = len(dom_class)
    return [float((dom_class == c).sum()) / n for c in range(N_CLASSES)]


# ── Main experiment loop ──────────────────────────────────────────────────────

def run_experiment(cfg: dict) -> None:
    print("=" * 72)
    print("  IMBALANCED DIET TOPOGRAPHIC FMNIST EXPERIMENT")
    print("=" * 72)
    print(json.dumps({k: v for k, v in cfg.items() if k != "variants"}, indent=2))

    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}\n")

    out_dir  = Path(cfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    res_dir  = out_dir / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    full_train = datasets.FashionMNIST(
        cfg["data_dir"], train=True,  download=True, transform=transform)
    full_val   = datasets.FashionMNIST(
        cfg["data_dir"], train=False, download=True, transform=transform)

    print("Extracting 4-class splits …")
    train_per_class = _extract_4class_splits(full_train)
    val_per_class   = _extract_4class_splits(full_val)

    for c in range(N_CLASSES):
        print(f"  {CLASS_NAMES[c]:8s}  train={len(train_per_class[c]):5,d}"
              f"  val={len(val_per_class[c]):5,d}")

    val_ds     = build_balanced_val_dataset(val_per_class)
    pin_memory = device.startswith("cuda")
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=2, pin_memory=pin_memory,
    )

    all_results: dict = {}

    for diet_name, proportions in DIETS.items():
        print(f"\n{'=' * 72}")
        print(f"  DIET: {diet_name}   ({proportions})")
        print(f"{'=' * 72}")

        train_ds = build_diet_dataset(
            train_per_class,
            proportions,
            total_samples=cfg["total_samples"],
            seed=cfg["seed"],
        )
        n_per_cls = [round(cfg["total_samples"] * p) for p in proportions]
        print(f"  Training samples: {[f'{CLASS_NAMES[c]}={n}' for c, n in enumerate(n_per_cls)]}")
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=2, pin_memory=pin_memory,
        )

        for variant_name, vcfg in VARIANTS.items():
            key = f"{diet_name}__{variant_name}"
            print(f"\n── {variant_name}  ({diet_name}) ──")

            torch.manual_seed(cfg["seed"])
            np.random.seed(cfg["seed"])
            model = SimpleNN4(hidden_size=cfg["hidden_size"]).to(device)

            history = train_one_run(
                model       = model,
                train_loader= train_loader,
                val_loader  = val_loader,
                variant_cfg = vcfg,
                epochs      = cfg["epochs"],
                lr          = cfg["lr"],
                device      = device,
                print_freq  = cfg["print_freq"],
                label       = f"{variant_name}|{diet_name}",
            )

            # ── Save checkpoint ────────────────────────────────────────────
            ckpt_path = ckpt_dir / f"model_{key}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "diet":        diet_name,
                "variant":     variant_name,
                "hidden_size": cfg["hidden_size"],
            }, ckpt_path)

            # ── Post-training analysis ─────────────────────────────────────
            print(f"  Computing selectivity statistics …")
            t_stats   = compute_t_stats(model, val_loader, device)
            act_hm    = compute_mean_activation_heatmap(model, val_loader, device)
            cls_hm    = compute_class_activation_heatmaps(model, val_loader, device)
            scs       = compute_spatial_clustering_score(t_stats, cfg["hidden_size"])
            territory = compute_class_territories(t_stats)

            # Mean normalised activation entropy over the balanced val set
            model.eval()
            all_ent_list: list[float] = []
            with torch.no_grad():
                for imgs, _ in val_loader:
                    imgs = imgs.to(device)
                    model(imgs)
                    all_ent_list.append(_activation_entropy(model._fc1_acts))
            act_ent = float(np.mean(all_ent_list))

            print(f"  SCS={scs:.4f}  ActEnt={act_ent:.4f}  "
                  f"Territory: " +
                  "  ".join(f"{CLASS_NAMES[c]}={territory[c]:.2f}"
                            for c in range(N_CLASSES)))

            all_results[key] = {
                "diet":             diet_name,
                "variant":          variant_name,
                "proportions":      proportions,
                "n_samples_per_class": n_per_cls,
                "history":          history,
                "final_val_acc":    history["val_acc"][-1],
                "final_val_acc_per_class": history["val_acc_per_class"][-1],
                # Analysis arrays stored as nested lists for JSON portability
                "t_stats":          t_stats.tolist(),           # (n_neurons, 4)
                "activation_heatmap": act_hm.tolist(),          # (n_neurons,)
                "class_activation_heatmaps": cls_hm.tolist(),   # (4, n_neurons)
                "spatial_clustering_score":  scs,
                "class_territories":         territory,          # [f0, f1, f2, f3]
                "activation_entropy":        act_ent,
                "checkpoint_path":           str(ckpt_path),
                "hidden_size":               cfg["hidden_size"],
            }

    # ── Save results ───────────────────────────────────────────────────────────
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_path   = res_dir / f"imbalanced_diet_results_{ts}.json"
    last_path = res_dir / "imbalanced_diet_results_latest.json"

    for path in (ts_path, last_path):
        with open(path, "w") as fh:
            json.dump(all_results, fh, indent=2)
    print(f"\nResults saved to:\n  {last_path}")

    # Quick summary table
    print("\nFinal validation accuracy summary:")
    print(f"  {'Key':45s}  {'ValAcc':>7s}  "
          + "  ".join(f"{n[:4]:>6s}" for n in CLASS_NAMES))
    print("  " + "-" * 75)
    for key, r in all_results.items():
        cls_str = "  ".join(f"{v:6.1f}" for v in r["final_val_acc_per_class"])
        print(f"  {key:45s}  {r['final_val_acc']:6.1f}%  {cls_str}")


# ── Config & entry point ──────────────────────────────────────────────────────

DEFAULT_CFG: dict = {
    "data_dir":      None,
    "output_dir":    None,
    "hidden_size":   256,
    "epochs":        30,
    "total_samples": 10_000,
    "batch_size":    128,
    "lr":            5e-4,
    "device":        "cuda:0",
    "print_freq":    5,
    "seed":          42,
}


def get_config() -> dict:
    p = argparse.ArgumentParser(description="Imbalanced diet topographic FashionMNIST")
    p.add_argument("--config",         default=None)
    p.add_argument("--data-dir",       default=None)
    p.add_argument("--output-dir",     default=None)
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--total-samples",  type=int,   default=None)
    p.add_argument("--batch-size",     type=int,   default=None)
    p.add_argument("--lr",             type=float, default=None)
    p.add_argument("--device",         default=None)
    p.add_argument("--print-freq",     type=int,   default=None)
    p.add_argument("--seed",           type=int,   default=None)
    cli = p.parse_args()

    cfg = copy.deepcopy(DEFAULT_CFG)

    default_json = str(BASE_DIR / "configs" / "imbalanced_diet.json")
    config_path  = cli.config or (default_json if Path(default_json).exists() else None)
    if config_path:
        with open(config_path) as fh:
            for k, v in json.load(fh).items():
                if not k.startswith("_") and v is not None:
                    cfg[k] = v

    overrides = {
        "data_dir":      cli.data_dir,
        "output_dir":    cli.output_dir,
        "epochs":        cli.epochs,
        "total_samples": cli.total_samples,
        "batch_size":    cli.batch_size,
        "lr":            cli.lr,
        "device":        cli.device,
        "print_freq":    cli.print_freq,
        "seed":          cli.seed,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    if not cfg["data_dir"]:
        cfg["data_dir"] = str(BASE_DIR / "data")
    if not cfg["output_dir"]:
        cfg["output_dir"] = str(OUTPUT_DIR)

    return cfg


if __name__ == "__main__":
    cfg = get_config()
    run_experiment(cfg)
