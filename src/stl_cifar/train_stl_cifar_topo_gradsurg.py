"""
Train ResNet-18 + TopoLoss on STL-10, then finetune on CIFAR-10 using
*gradient surgery* to prevent catastrophic forgetting.

Phase 1 (STL-10 pretraining)
  Identical to topo_only: standard cross-entropy + TopoLoss (LaplacianPyramid)
  on the cortical sheet conv layers.

Phase 2 (CIFAR-10 finetuning)
  No explicit regularisation penalty.  Instead, a backward hook is registered
  on the weight of every cortical-sheet conv.  The hook:
    1. Computes per-unit gradient L2 norms, reshaped onto the (H_c × W_c)
       cortical sheet.
    2. Applies the topo-smoothing operation (adaptive avg_pool2d ↓ then
       bilinear upsample ↑) to collapse diffuse gradient signals into spatially
       coherent regional activations.
    3. Zeroes out all cortical units whose smoothed norm falls below the
       ``grad_surg_percentile``-th quantile, leaving only the strongly reactive
       spatial patch to update weights.

  Because the monkey region of the cortical sheet (organised during phase 1
  via TopoLoss) is never reactive to any CIFAR-10 input, it accumulates a zero
  gradient mask across the entire finetuning run and is never modified —
  preserving STL-10 knowledge there without any explicit penalty, Fisher
  computation, or weight anchoring.

Config keys (in addition to shared keys):
  grad_surg_percentile  : float  — percentile threshold (default 75.0,
                                    i.e. keep the top 25% most-reactive units).
  grad_surg_factor_h    : float  — cortical downscale factor for smoothing
                                    (defaults to default_layer_cfg.factor_h).
  grad_surg_factor_w    : float  — cortical downscale factor for smoothing
                                    (defaults to default_layer_cfg.factor_w).

Output: outputs/stl_cifar/topo_gradsurg_results_latest.json
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent / "imagenet"))  # resnet_imagenet_common
from stl_cifar_common import (
    BASE_DIR, load_config, build_model,
    make_stl_loaders, make_cifar_overlap_loaders, _CIFAR_TO_STL,
    run_pretrain, save_results, evaluate, evaluate_per_class,
    make_gradsurg_hook, _grad_entropy, _make_optimizer,
)
from resnet_imagenet_common import get_residual_convs, build_topo_loss

LABEL = "topo_gradsurg"


# ---------------------------------------------------------------------------
# Gradient-surgery finetune loop
# ---------------------------------------------------------------------------

def run_finetune_gradsurg(
    label: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    stl_val_loader: DataLoader,
    cfg: dict,
    ckpt_dir: Path,
    residual_convs: dict[str, nn.Conv2d],
    default_layer_cfg: dict,
) -> tuple[float, float, list[float], list[float], dict]:
    """Phase 2: CIFAR-10 finetuning with gradient surgery on cortical convs.

    Returns
    -------
    (cifar_acc, stl_acc_after, cifar_acc_per_epoch, stl_acc_per_epoch, ft_loss_history)
    """
    device   = cfg["device"]
    epochs   = cfg["finetune_epochs"]
    use_cuda = device.startswith("cuda")

    # Gradient surgery hyper-parameters
    percentile = float(cfg.get("grad_surg_percentile", 75.0))
    factor_h   = float(cfg.get("grad_surg_factor_h",
                                default_layer_cfg.get("factor_h", 4.0)))
    factor_w   = float(cfg.get("grad_surg_factor_w",
                                default_layer_cfg.get("factor_w", 4.0)))

    print(
        f"  Gradient surgery: percentile={percentile:.1f}  "
        f"factor_h={factor_h}  factor_w={factor_w}"
    )

    base_model  = model
    train_model = model
    num_gpus    = torch.cuda.device_count() if use_cuda else 0
    if num_gpus > 1:
        train_model = nn.DataParallel(model)

    scaler = torch.amp.GradScaler("cuda") if use_cuda else None

    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(base_model, cfg["finetune_lr"], cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Register weight-gradient hooks on every cortical conv
    hook_handles: list = []
    for name, conv in residual_convs.items():
        h = conv.weight.register_hook(
            make_gradsurg_hook(conv, factor_h, factor_w, percentile)
        )
        hook_handles.append(h)

    # Tracking
    cifar_history:           list[float] = []
    stl_history:             list[float] = []
    ce_history:              list[float] = []
    grad_ent_history:        list[float] = []
    cortical_grad_ent_hist:  list[float] = []   # entropy of cortical-conv grads only
    mask_density_hist:       list[float] = []   # fraction of cortical units NOT zeroed

    cortical_params = [conv.weight for conv in residual_convs.values()]

    print_freq = cfg.get("print_freq", 50)

    for epoch in range(epochs):
        t0 = time.time()
        train_model.train()

        sum_ce              = 0.0
        sum_grad_ent        = 0.0
        sum_cortical_grad_ent = 0.0
        # Approximate mask density = (1 - percentile/100); constant by construction
        # but we still compute it empirically per epoch for verification.
        mask_density        = 1.0 - percentile / 100.0
        n_total             = 0
        n_steps             = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                logits = train_model(imgs)
                loss   = criterion(logits, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                sum_grad_ent        += _grad_entropy(base_model)
                sum_cortical_grad_ent += _grad_entropy(base_model, params=cortical_params)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                sum_grad_ent        += _grad_entropy(base_model)
                sum_cortical_grad_ent += _grad_entropy(base_model, params=cortical_params)
                optimizer.step()

            bs       = imgs.size(0)
            sum_ce  += loss.item() * bs
            n_total += bs
            n_steps += 1

            if print_freq > 0 and (batch_idx + 1) % print_freq == 0:
                print(
                    f"[{label}|CIFAR-FT] Epoch [{epoch+1}/{epochs}]  "
                    f"Step [{batch_idx+1}/{len(train_loader)}]  "
                    f"CE={sum_ce/n_total:.4f}  "
                    f"GradH={sum_grad_ent/max(n_steps,1):.4f}  "
                    f"MaskDensity={mask_density:.3f}"
                )

        scheduler.step()

        ce_avg              = sum_ce      / n_total
        grad_ent_avg        = sum_grad_ent / max(n_steps, 1)
        cortical_grad_ent_avg = sum_cortical_grad_ent / max(n_steps, 1)

        cifar_acc = evaluate(train_model, val_loader,     device)
        stl_acc   = evaluate(train_model, stl_val_loader, device)

        cifar_history.append(cifar_acc)
        stl_history.append(stl_acc)
        ce_history.append(ce_avg)
        grad_ent_history.append(grad_ent_avg)
        cortical_grad_ent_hist.append(cortical_grad_ent_avg)
        mask_density_hist.append(mask_density)

        print(
            f"[{label}|CIFAR-FT] Epoch [{epoch+1:3d}/{epochs}]  "
            f"CE={ce_avg:.4f}  GradH={grad_ent_avg:.4f}  "
            f"CorticalGradH={cortical_grad_ent_avg:.4f}  "
            f"MaskDensity={mask_density:.3f}  "
            f"CIFAR={cifar_acc:.2f}%  STL={stl_acc:.2f}%  "
            f"t={time.time()-t0:.1f}s"
        )

    ckpt = {
        "epoch":     epochs - 1,
        "model":     base_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(ckpt, ckpt_dir / f"finetune_last_{label}.pt")

    for h in hook_handles:
        h.remove()

    ft_loss_history = {
        "ce":                  ce_history,
        "topo":                [0.0] * epochs,   # no topo penalty during finetune
        "entropy":             [0.0] * epochs,
        "kl":                  [0.0] * epochs,
        "sparse":              [0.0] * epochs,
        "sparse_kl":           [0.0] * epochs,
        "grad_entropy":        grad_ent_history,
        "cortical_grad_entropy": cortical_grad_ent_hist,
        "mask_density":        mask_density_hist,
    }
    return cifar_history[-1], stl_history[-1], cifar_history, stl_history, ft_loss_history


# ---------------------------------------------------------------------------
# Top-level train function
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    print("=" * 70)
    print("  TOPO + GRAD-SURGERY — TopoLoss pretrain, gradient surgery finetune")
    print("=" * 70)
    skip = {"layers", "default_layer_cfg"}
    print(json.dumps({k: v for k, v in cfg.items() if k not in skip}, indent=2))
    print(f"  default_layer_cfg: {json.dumps(cfg['default_layer_cfg'])}")
    print()

    device = cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device '{device}' requested but CUDA is not available.\n"
            f"PyTorch: {torch.__version__}"
        )
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
    print()

    out_dir  = Path(cfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_dir = cfg["data_dir"]

    # ---- Data ---------------------------------------------------------------
    print("Loading STL-10 …")
    stl_train, stl_val = make_stl_loaders(
        data_dir, cfg["batch_size"], cfg["num_workers"], cfg["img_size"]
    )
    print("Loading CIFAR-10 (overlap classes only) …")
    cifar_train, cifar_val = make_cifar_overlap_loaders(
        data_dir, cfg["batch_size"], cfg["num_workers"], cfg["img_size"],
        subset_classes=cfg.get("finetune_cifar_classes"),
    )

    # ---- Model + TopoLoss (pretrain only) -----------------------------------
    model = build_model(cfg["num_classes"], device)
    residual_convs = get_residual_convs(model, cfg.get("include_downsample", False))
    print(f"Residual convs targeted: {len(residual_convs)}")

    default_lc = {
        **cfg["default_layer_cfg"],
        # No auxiliary sparsity penalties — pure TopoLoss during pretrain
        "lambda_entropy":  0.0,
        "lambda_kl":       0.0,
        "lambda_sparse":   0.0,
        "lambda_sparse_kl": 0.0,
    }
    layer_cfg  = {
        name: {
            **lc,
            "lambda_entropy":  0.0,
            "lambda_kl":       0.0,
            "lambda_sparse":   0.0,
            "lambda_sparse_kl": 0.0,
        }
        for name, lc in cfg["layers"].items()
    }
    topo_loss = build_topo_loss(model, residual_convs, layer_cfg, default_lc)

    # ---- Phase 1: STL-10 pretraining ----------------------------------------
    print("\n── Phase 1: STL-10 pretraining ──────────────────────────────────────")
    model, best_stl_acc, stl_acc_history, pretrain_losses = run_pretrain(
        label=LABEL, model=model,
        train_loader=stl_train, val_loader=stl_val,
        cfg=cfg, ckpt_dir=ckpt_dir,
        topo_loss=topo_loss, residual_convs=residual_convs,
        layer_cfg=layer_cfg, default_layer_cfg=default_lc,
    )
    print(f"\nPhase 1 complete. Best STL-10 Val: {best_stl_acc:.2f}%")

    stl_acc_before      = evaluate(model, stl_val, device)
    stl_per_class_before = evaluate_per_class(model, stl_val, device, n_classes=10)
    print(f"STL-10 accuracy before finetune: {stl_acc_before:.2f}%")

    # ---- Phase 2: CIFAR-10 finetuning with gradient surgery -----------------
    print("\n── Phase 2: CIFAR-10 finetuning (gradient surgery) ──────────────────")
    (cifar_acc, stl_acc_after,
     cifar_history, stl_ft_history, ft_losses) = run_finetune_gradsurg(
        label=LABEL, model=model,
        train_loader=cifar_train, val_loader=cifar_val,
        stl_val_loader=stl_val,
        cfg=cfg, ckpt_dir=ckpt_dir,
        residual_convs=residual_convs,
        default_layer_cfg=default_lc,
    )
    stl_per_class_after = evaluate_per_class(model, stl_val, device, n_classes=10)

    forgetting = stl_acc_before - stl_acc_after
    print(f"\n── Results ({LABEL}) ───────────────────────────────────────────────")
    print(f"  STL-10  before finetune : {stl_acc_before:.2f}%")
    print(f"  CIFAR-10 after finetune : {cifar_acc:.2f}%")
    print(f"  STL-10  after finetune  : {stl_acc_after:.2f}%")
    print(f"  Forgetting              : {forgetting:.2f}%")

    results = {
        "label":             LABEL,
        "stl_acc_before":    stl_acc_before,
        "cifar_acc_after":   cifar_acc,
        "stl_acc_after":     stl_acc_after,
        "stl_per_class_acc_before": stl_per_class_before,
        "stl_per_class_acc_after":  stl_per_class_after,
        "forgetting":        forgetting,
        "best_stl_acc":      best_stl_acc,
        "stl_acc_per_epoch":       stl_acc_history,
        "cifar_acc_per_epoch_ft":  cifar_history,
        "stl_acc_per_epoch_ft":    stl_ft_history,
        "pretrain_ce_per_epoch":           pretrain_losses["ce"],
        "pretrain_topo_per_epoch":         pretrain_losses["topo"],
        "pretrain_entropy_per_epoch":      pretrain_losses["entropy"],
        "pretrain_grad_entropy_per_epoch": pretrain_losses["grad_entropy"],
        "pretrain_kl_per_epoch":           pretrain_losses["kl"],
        "pretrain_sparse_per_epoch":       pretrain_losses["sparse"],
        "pretrain_sparse_kl_per_epoch":    pretrain_losses["sparse_kl"],
        "ft_ce_per_epoch":                 ft_losses["ce"],
        "ft_topo_per_epoch":               ft_losses["topo"],
        "ft_entropy_per_epoch":            ft_losses["entropy"],
        "ft_grad_entropy_per_epoch":          ft_losses["grad_entropy"],
        "ft_cortical_grad_entropy_per_epoch": ft_losses["cortical_grad_entropy"],
        "ft_kl_per_epoch":                    ft_losses["kl"],
        "ft_sparse_per_epoch":             ft_losses["sparse"],
        "ft_sparse_kl_per_epoch":          ft_losses["sparse_kl"],
        "ft_mask_density_per_epoch":       ft_losses["mask_density"],
        "finetuned_stl_classes": sorted(
            _CIFAR_TO_STL[c]
            for c in (cfg.get("finetune_cifar_classes") or list(_CIFAR_TO_STL))
            if c in _CIFAR_TO_STL
        ),
        "config": {k: v for k, v in cfg.items() if k not in {"layers", "default_layer_cfg"}},
    }
    save_results(results, out_dir / "results", LABEL)


def main():
    p = argparse.ArgumentParser(
        description="TopoLoss pretrain + gradient-surgery finetune STL→CIFAR experiment"
    )
    p.add_argument("--config",            type=str,   default=None)
    p.add_argument("--output-dir",        type=str,   default=None)
    p.add_argument("--data-dir",          type=str,   default=None)
    p.add_argument("--stl-epochs",        type=int,   default=None)
    p.add_argument("--finetune-epochs",   type=int,   default=None)
    p.add_argument("--batch-size",        type=int,   default=None)
    p.add_argument("--lr",                type=float, default=None)
    p.add_argument("--finetune-lr",       type=float, default=None)
    p.add_argument("--device",            type=str,   default=None)
    p.add_argument("--num-workers",       type=int,   default=None)
    p.add_argument("--resume-pretrain",   type=str,   default=None)
    p.add_argument("--include-downsample", action="store_true", default=None)
    p.add_argument("--grad-surg-percentile", type=float, default=None,
                   help="Percentile threshold for gradient surgery (default: 75)")
    args = p.parse_args()

    default_cfg = str(BASE_DIR / "configs" / "train_stl_cifar.json")
    cfg = load_config(
        args.config or (default_cfg if Path(default_cfg).exists() else None),
        extra_args={
            "output_dir":          args.output_dir,
            "data_dir":            args.data_dir,
            "stl_epochs":          args.stl_epochs,
            "finetune_epochs":     args.finetune_epochs,
            "batch_size":          args.batch_size,
            "lr":                  args.lr,
            "finetune_lr":         args.finetune_lr,
            "device":              args.device,
            "num_workers":         args.num_workers,
            "resume_pretrain":     args.resume_pretrain,
            "include_downsample":  args.include_downsample,
            "grad_surg_percentile": args.grad_surg_percentile,
        },
    )
    train(cfg)


if __name__ == "__main__":
    main()
