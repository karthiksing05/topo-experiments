"""
Train ResNet-18 (baseline) on STL-10, then finetune on CIFAR-10.
Measures catastrophic forgetting with CrossEntropy loss only.

Output: outputs/stl_cifar/baseline_results_latest.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stl_cifar_common import (
    BASE_DIR, load_config, build_model,
    make_stl_loaders, make_cifar_overlap_loaders, _CIFAR_TO_STL,
    run_pretrain, run_finetune, save_results,
)

LABEL = "baseline"


def train(cfg: dict) -> None:
    print("=" * 70)
    print("  BASELINE — CrossEntropy only (STL-10 → CIFAR-10)")
    print("=" * 70)
    skip = {"layers", "default_layer_cfg"}
    print(json.dumps({k: v for k, v in cfg.items() if k not in skip}, indent=2))
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

    # ---- Model --------------------------------------------------------------
    model = build_model(cfg["num_classes"], device)

    # ---- Phase 1: Pretrain on STL-10 ----------------------------------------
    print("\n── Phase 1: STL-10 pretraining ──────────────────────────────────────")
    model, best_stl_acc, stl_acc_history, pretrain_losses = run_pretrain(
        label=LABEL, model=model,
        train_loader=stl_train, val_loader=stl_val,
        cfg=cfg, ckpt_dir=ckpt_dir,
        topo_loss=None, residual_convs={},
        layer_cfg={}, default_layer_cfg=cfg["default_layer_cfg"],
    )
    print(f"\nPhase 1 complete. Best STL-10 Val: {best_stl_acc:.2f}%")

    # Evaluate STL *before* finetuning (use best weights already loaded)
    from stl_cifar_common import evaluate, evaluate_per_class
    stl_acc_before = evaluate(model, stl_val, device)
    stl_per_class_before = evaluate_per_class(model, stl_val, device, n_classes=10)
    print(f"STL-10 accuracy before finetune: {stl_acc_before:.2f}%")

    # ---- Phase 2: Finetune on CIFAR-10 --------------------------------------
    print("\n── Phase 2: CIFAR-10 finetuning ─────────────────────────────────────")
    cifar_acc, stl_acc_after, cifar_history, stl_ft_history, ft_losses = run_finetune(
        label=LABEL, model=model,
        train_loader=cifar_train, val_loader=cifar_val,
        stl_val_loader=stl_val,
        cfg=cfg, ckpt_dir=ckpt_dir,
        topo_loss=None, residual_convs={},
        layer_cfg={}, default_layer_cfg=cfg["default_layer_cfg"],
    )
    stl_per_class_after = evaluate_per_class(model, stl_val, device, n_classes=10)

    forgetting = stl_acc_before - stl_acc_after
    print(f"\n── Results ({LABEL}) ──────────────────────────────────────────────────")
    print(f"  STL-10  before finetune : {stl_acc_before:.2f}%")
    print(f"  CIFAR-10 after finetune : {cifar_acc:.2f}%")
    print(f"  STL-10  after finetune  : {stl_acc_after:.2f}%")
    print(f"  Forgetting              : {forgetting:.2f}%")

    results = {
        "label":            LABEL,
        "stl_acc_before":   stl_acc_before,
        "cifar_acc_after":  cifar_acc,
        "stl_acc_after":    stl_acc_after,
        "stl_per_class_acc_before": stl_per_class_before,
        "stl_per_class_acc_after":  stl_per_class_after,
        "forgetting":       forgetting,
        "best_stl_acc":     best_stl_acc,
        "stl_acc_per_epoch":      stl_acc_history,
        "cifar_acc_per_epoch_ft": cifar_history,
        "stl_acc_per_epoch_ft":   stl_ft_history,
        "pretrain_ce_per_epoch":           pretrain_losses["ce"],
        "pretrain_topo_per_epoch":         pretrain_losses["topo"],
        "pretrain_entropy_per_epoch":      pretrain_losses["entropy"],
        "pretrain_grad_entropy_per_epoch": pretrain_losses["grad_entropy"],
        "pretrain_kl_per_epoch":           pretrain_losses["kl"],
        "pretrain_sparse_per_epoch":        pretrain_losses["sparse"],
        "ft_ce_per_epoch":                  ft_losses["ce"],
        "ft_topo_per_epoch":               ft_losses["topo"],
        "ft_entropy_per_epoch":            ft_losses["entropy"],
        "ft_grad_entropy_per_epoch":       ft_losses["grad_entropy"],
        "ft_kl_per_epoch":                 ft_losses["kl"],
        "ft_sparse_per_epoch":             ft_losses["sparse"],
        "finetuned_stl_classes":  sorted(
            _CIFAR_TO_STL[c] for c in (cfg.get("finetune_cifar_classes") or list(_CIFAR_TO_STL))
            if c in _CIFAR_TO_STL
        ),
        "config": {k: v for k, v in cfg.items() if k not in {"layers", "default_layer_cfg"}},
    }
    save_results(results, out_dir / "results", LABEL)


def main():
    p = argparse.ArgumentParser(description="Baseline STL→CIFAR forgetting experiment")
    p.add_argument("--config",          type=str, default=None)
    p.add_argument("--output-dir",      type=str, default=None)
    p.add_argument("--data-dir",        type=str, default=None)
    p.add_argument("--stl-epochs",      type=int, default=None)
    p.add_argument("--finetune-epochs", type=int, default=None)
    p.add_argument("--batch-size",      type=int, default=None)
    p.add_argument("--lr",              type=float, default=None)
    p.add_argument("--finetune-lr",     type=float, default=None)
    p.add_argument("--device",          type=str, default=None)
    p.add_argument("--num-workers",     type=int, default=None)
    p.add_argument("--resume-pretrain", type=str, default=None)
    args = p.parse_args()

    default_cfg = str(BASE_DIR / "configs" / "train_stl_cifar.json")
    cfg = load_config(
        args.config or (default_cfg if Path(default_cfg).exists() else None),
        extra_args={
            "output_dir":      args.output_dir,
            "data_dir":        args.data_dir,
            "stl_epochs":      args.stl_epochs,
            "finetune_epochs": args.finetune_epochs,
            "batch_size":      args.batch_size,
            "lr":              args.lr,
            "finetune_lr":     args.finetune_lr,
            "device":          args.device,
            "num_workers":     args.num_workers,
            "resume_pretrain": args.resume_pretrain,
        },
    )
    train(cfg)


if __name__ == "__main__":
    main()
