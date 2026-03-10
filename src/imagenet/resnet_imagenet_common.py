"""
Shared utilities for ResNet-18 / ImageNet training scripts.

Provides:
  - HFImageNetRawDataset  : HF → raw PIL images (for FFCV writer)
  - make_ffcv_loaders     : build FFCV Loader objects for train/val
  - residual conv helpers : _is_residual_conv, get_residual_convs
  - cortical_entropy_loss
  - build_topo_loss
  - run_training          : epoch loop (returns model with best weights loaded)
  - evaluate              : top-1 / top-5 validation
  - get_base_config / get_config_for_variant : config parsing helpers
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tv_models
from torch.utils.data import Dataset

# FFCV
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import (
    IntDecoder,
    RandomResizedCropRGBImageDecoder,
    CenterCropRGBImageDecoder,
)
from ffcv.transforms import (
    ToTensor,
    ToDevice,
    Squeeze,
    NormalizeImage,
    RandomHorizontalFlip,
    ToTorchImage,
)

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

# ImageNet normalisation constants (values in [0, 255] for NormalizeImage)
IMAGENET_MEAN = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32)
IMAGENET_STD  = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32)

# ---------------------------------------------------------------------------
# HuggingFace raw dataset (no transform — used only by FFCV writer)
# ---------------------------------------------------------------------------

class HFImageNetRawDataset(Dataset):
    """Returns (PIL.Image [RGB], int label) directly from HF Hub.

    No transform is applied; the FFCV DatasetWriter handles encoding.

    Parameters
    ----------
    split : ``"train"`` or ``"validation"``.
    token : HF access token (falls back to HF_TOKEN env var or cached login).
    """

    def __init__(self, split: str, token: str | None = None) -> None:
        if not _HF_DATASETS_AVAILABLE:
            raise ImportError(
                "The 'datasets' package is required.\n"
                "Install it with:  pip install datasets"
            )
        _token = token or os.environ.get("HF_TOKEN") or True
        print(f"  Loading ILSVRC/imagenet-1k split='{split}' from HuggingFace Hub ...")
        self._ds = hf_datasets.load_dataset(
            "ILSVRC/imagenet-1k",
            split=split,
            token=_token,
            trust_remote_code=True,
        )
        print(f"  Loaded {len(self._ds):,} examples.")

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        example = self._ds[int(idx)]
        img = example["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img, example["label"]


# ---------------------------------------------------------------------------
# FFCV data loaders
# ---------------------------------------------------------------------------

def make_ffcv_loaders(
    train_beton: str,
    val_beton: str,
    batch_size: int,
    num_workers: int,
    device: str,
    distributed: bool = False,
) -> tuple:
    """Return ``(train_loader, val_loader)`` backed by FFCV.

    Images are decoded, augmented, and transferred to *device* inside the
    FFCV pipeline (non-blocking).  The loaders yield ``(images, labels)``
    tensors in the standard ``(B, C, H, W)`` / ``(B,)`` format.

    Parameters
    ----------
    train_beton : path to the train .beton file.
    val_beton   : path to the val .beton file.
    batch_size  : mini-batch size (train).  Val uses batch_size * 2.
    num_workers : number of FFCV background loader threads.
    device      : target device string (e.g. ``"cuda:0"``).
    distributed : use DistributedSampler ordering if True.
    """
    _dev = torch.device(device)

    train_image_pipeline = [
        RandomResizedCropRGBImageDecoder((224, 224)),
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(_dev, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
    ]
    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(_dev, non_blocking=True),
    ]
    val_image_pipeline = [
        CenterCropRGBImageDecoder((224, 224), ratio=224 / 256),
        ToTensor(),
        ToDevice(_dev, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
    ]

    train_order = (
        OrderOption.RANDOM
        if not distributed
        else OrderOption.RANDOM  # replace with DISTRIBUTED when using DDP
    )

    train_loader = Loader(
        train_beton,
        batch_size=batch_size,
        num_workers=num_workers,
        order=train_order,
        os_cache=True,
        drop_last=True,
        pipelines={"image": train_image_pipeline, "label": label_pipeline},
    )
    val_loader = Loader(
        val_beton,
        batch_size=batch_size * 2,
        num_workers=num_workers,
        order=OrderOption.SEQUENTIAL,
        os_cache=True,
        drop_last=False,
        pipelines={"image": val_image_pipeline, "label": label_pipeline},
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Residual-block conv identification
# ---------------------------------------------------------------------------

_RESIDUAL_GROUPS = ("layer1", "layer2", "layer3", "layer4")


def _is_residual_conv(name: str, include_downsample: bool) -> bool:
    if not any(name.startswith(g) for g in _RESIDUAL_GROUPS):
        return False
    if "downsample" in name and not include_downsample:
        return False
    return True


def get_residual_convs(
    model: nn.Module, include_downsample: bool = False
) -> dict[str, nn.Conv2d]:
    """Return ``{module_name: Conv2d}`` for every targeted residual-block conv."""
    result: dict[str, nn.Conv2d] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and _is_residual_conv(name, include_downsample):
            result[name] = module
    return result


# ---------------------------------------------------------------------------
# Entropy-sparsity loss
# ---------------------------------------------------------------------------

def cortical_entropy_loss(
    activations: torch.Tensor,
    factor_h: float,
    factor_w: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-sample entropy on the downsampled cortical sheet.

    Global-avg-pools conv activations (B, C, H, W) → (B, C), reshapes into
    the cortical sheet, downsamples, applies softmax, then returns mean
    entropy (lower = sparser).
    """
    if activations.ndim == 4:
        activations = activations.mean(dim=(2, 3))   # (B, C)

    B, N  = activations.shape
    size  = find_cortical_sheet_size(N)
    H, W  = size.height, size.width

    sheet  = activations[:, : H * W].reshape(B, 1, H, W)
    H_d    = max(1, round(H / factor_h))
    W_d    = max(1, round(W / factor_w))
    flat   = F.adaptive_avg_pool2d(sheet, (H_d, W_d)).reshape(B, -1)

    probs   = F.softmax(flat / temperature, dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
    return entropy


# ---------------------------------------------------------------------------
# Build TopoLoss
# ---------------------------------------------------------------------------

def build_topo_loss(
    model: nn.Module,
    residual_convs: dict[str, nn.Conv2d],
    layer_cfg: dict,
    default_cfg: dict,
) -> TopoLoss:
    """One ``LaplacianPyramid`` per targeted conv → ``TopoLoss``."""
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
    train_loader,
    val_loader,
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
    """Train *model* and return it loaded with the best-checkpoint weights.

    Automatically uses:
      - AMP (mixed precision) on CUDA for ~2× throughput
      - DataParallel when multiple GPUs are visible
    """
    use_cuda = device.startswith("cuda")

    # ---- Multi-GPU DataParallel ----------------------------------------
    # Keep a reference to the base (unwrapped) model for topo_loss and
    # checkpoint saving; use train_model for the forward pass.
    base_model  = model
    train_model = model
    num_gpus    = torch.cuda.device_count() if use_cuda else 0
    if num_gpus > 1:
        print(f"  DataParallel: using {num_gpus} GPUs")
        train_model = nn.DataParallel(model)

    # ---- AMP scaler ----------------------------------------------------
    scaler = torch.amp.GradScaler("cuda") if use_cuda else None
    if scaler is not None:
        print(f"  AMP enabled (GradScaler)")

    # ---- Forward hooks for entropy loss (on base_model convs) ----------
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

    ckpt: dict = {}  # will be set inside the loop

    for epoch in range(start_epoch, epochs):
        train_model.train()

        sum_ce = sum_topo = sum_entropy = 0.0
        n_correct = n_total = 0
        t0 = time.time()

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            # FFCV already places tensors on device; enforce dtype
            imgs = imgs.float()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                logits = train_model(imgs)
                ce     = criterion(logits, labels)

                extra = torch.zeros(1, device=imgs.device)

                if topo_loss is not None:
                    # Use base_model so TopoLoss can locate its registered layers.
                    # Move result to primary device: DataParallel may leave tensors
                    # on a replica GPU (e.g. cuda:1) while extra lives on cuda:0.
                    primary = extra.device
                    topo  = topo_loss.compute(model=base_model, reduce_mean=True).to(primary)
                    extra = extra + topo
                    sum_topo += topo.item() * imgs.size(0)

                    for name in residual_convs:
                        act = act_store[name]
                        if act is not None:
                            # Hook may have fired from a DataParallel replica GPU
                            act = act.to(primary)
                            lc  = {**default_layer_cfg, **layer_cfg.get(name, {})}
                            lam = lc.get("lambda_entropy", 0.0)
                            if lam > 0.0:
                                ent   = cortical_entropy_loss(
                                    act, lc["factor_h"], lc["factor_w"],
                                    lc.get("temperature", 1.0),
                                )
                                extra        = extra + lam * ent
                                sum_entropy += ent.item() * imgs.size(0)

                loss = ce + extra

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            bs = imgs.size(0)
            sum_ce    += ce.item() * bs
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_total   += bs

            if print_freq > 0 and (batch_idx + 1) % print_freq == 0:
                print(
                    f"[{label}] Epoch [{epoch+1}/{epochs}]  "
                    f"Step [{batch_idx+1}/{len(train_loader)}]  "
                    f"CE={sum_ce/n_total:.4f}  "
                    f"Topo={sum_topo/n_total:.5f}  "
                    f"Ent={sum_entropy/n_total:.5f}  "
                    f"Train@1={100*n_correct/n_total:.2f}%"
                )

        scheduler.step()

        top1, top5 = evaluate(train_model, val_loader)
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

        is_best = top1 > best_acc1
        if is_best:
            best_acc1 = top1

        # Always save base_model state (not DataParallel wrapper)
        ckpt = {
            "epoch":     epoch,
            "model":     base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc1": best_acc1,
        }

        if is_best:
            torch.save(ckpt, ckpt_dir / f"best_{label}.pt")
            print(f"  *** New best Val@1 = {best_acc1:.2f}% — checkpoint saved ***")

        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            torch.save(ckpt, ckpt_dir / f"epoch{epoch+1:04d}_{label}.pt")

    if ckpt:
        torch.save(ckpt, ckpt_dir / f"last_{label}.pt")

    for h in hook_handles:
        h.remove()

    print(f"\n[{label}] Training complete.  Best Val@1: {best_acc1:.2f}%")

    best_path = ckpt_dir / f"best_{label}.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        base_model.load_state_dict(best_ckpt["model"])
    return base_model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader) -> tuple[float, float]:
    """Return (top-1 %, top-5 %) on *loader*."""
    model.eval()
    correct1 = correct5 = total = 0

    for imgs, labels in loader:
        imgs   = imgs.float()
        logits = model(imgs)

        correct1 += (logits.argmax(1) == labels).sum().item()

        _, top5_preds = logits.topk(5, dim=1)
        correct5 += top5_preds.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

        total += labels.size(0)

    model.train()
    return 100 * correct1 / total, 100 * correct5 / total


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict = {
    "output_dir":  None,
    "num_classes": 1000,

    "train_beton": None,   # path to train .beton file (written by write_imagenet_ffcv.py)
    "val_beton":   None,   # path to val   .beton file

    "epochs":        90,
    "batch_size":    512,   # 2× GPUs → 2× batch; lr scaled accordingly
    "lr":            0.2,   # linear scaling: 0.1 × (512/256)
    "momentum":      0.9,
    "weight_decay":  1e-4,
    "lr_milestones": [30, 60, 80],
    "lr_gamma":      0.1,

    "device":      "cuda",
    "num_workers": 12,
    "print_freq":  200,
    "save_freq":   5,

    "resume":      None,

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


def load_config(
    config_path: str | None,
    extra_args: dict | None = None,
) -> dict:
    """Load JSON config, apply *extra_args* overrides, resolve paths."""
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
            print(f"Config file not found ({p}), using built-in defaults.")

    if extra_args:
        for key, val in extra_args.items():
            if val is not None:
                cfg[key] = val

    # Resolve default paths
    if cfg["output_dir"] is None:
        cfg["output_dir"] = str(OUTPUT_DIR)
    if cfg["train_beton"] is None:
        cfg["train_beton"] = str(BASE_DIR / "data" / "imagenet_ffcv" / "train.beton")
    if cfg["val_beton"] is None:
        cfg["val_beton"] = str(BASE_DIR / "data" / "imagenet_ffcv" / "val.beton")

    cfg.setdefault("hf_token", None)
    return cfg


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Add all shared CLI arguments to *p*."""
    default_cfg = str(BASE_DIR / "configs" / "train_resnet_imagenet.json")
    p.add_argument("--config",       type=str, default=default_cfg)
    p.add_argument("--output-dir",   type=str, default=None)
    p.add_argument("--train-beton",  type=str, default=None,
                   help="Path to train .beton file")
    p.add_argument("--val-beton",    type=str, default=None,
                   help="Path to val .beton file")
    p.add_argument("--num-classes",  type=int, default=None)
    p.add_argument("--epochs",       type=int, default=None)
    p.add_argument("--batch-size",   type=int, default=None)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--momentum",     type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--device",       type=str, default=None)
    p.add_argument("--num-workers",  type=int, default=None)
    p.add_argument("--print-freq",   type=int, default=None)
    p.add_argument("--save-freq",    type=int, default=None)
    p.add_argument("--resume",       type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--include-downsample", action="store_true", default=None,
                   help="Also apply TopoLoss+entropy to 1×1 downsample convs")
    p.add_argument("--hf-token",     type=str, default=None)


def make_model(num_classes: int, device: str) -> nn.Module:
    m = tv_models.resnet18(weights=None, num_classes=num_classes)
    return m.to(device)


def make_optimizer(model: nn.Module, cfg: dict) -> tuple:
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


def resume_checkpoint(model, opt, sched, ckpt_path, device):
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
    start = ckpt["epoch"] + 1
    best  = ckpt.get("best_acc1", 0.0)
    print(f"  Resumed from epoch {start}  (best_acc1={best:.2f}%)")
    return start, best


# ---------------------------------------------------------------------------
# Tee logger (used by each training script's __main__)
# ---------------------------------------------------------------------------

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


def setup_logging(label: str) -> tuple[Path, object, object]:
    """Create timestamped log file; return (log_path, orig_stdout, orig_stderr)."""
    log_dir = BASE_DIR / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"train_resnet_{label}-{ts}.txt"
    return log_path, sys.stdout, sys.stderr
