"""
Train ResNet-18 on ImageNet with TopoLoss + entropy-sparsity penalty.

Variant: topo_sparsity
  Loss = CrossEntropy + TopoLoss (LaplacianPyramid on every targeted
         residual-block conv) + per-channel entropy-sparsity penalty.

Requires FFCV .beton files written by write_imagenet_ffcv.py.

Usage
-----
python src/train/train_resnet_topo_sparsity.py
python src/train/train_resnet_topo_sparsity.py \\
    --config configs/train_resnet_imagenet.json \\
    --train-beton data/imagenet_ffcv/train.beton \\
    --val-beton   data/imagenet_ffcv/val.beton   \\
    --epochs 90 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Common utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))
from resnet_imagenet_common import (
    BASE_DIR,
    _Tee,
    add_common_args,
    build_topo_loss,
    get_residual_convs,
    load_config,
    make_ffcv_loaders,
    make_model,
    make_optimizer,
    resume_checkpoint,
    run_training,
    setup_logging,
)

LABEL = "topo_sparsity"


def train(cfg: dict) -> None:
    # ---- Config summary -----------------------------------------------------
    print("=" * 70)
    print("  TOPO SPARSITY — TopoLoss + entropy-sparsity on all residual convs")
    print("=" * 70)
    skip = {"layers", "default_layer_cfg"}
    print(json.dumps({k: v for k, v in cfg.items() if k not in skip}, indent=2))
    print("  default_layer_cfg:", json.dumps(cfg["default_layer_cfg"]))
    print(f"  layers: {len(cfg['layers'])} per-layer overrides\n")

    # ---- Device -------------------------------------------------------------
    device = cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device '{device}' requested but torch.cuda.is_available() is False.\n"
            f"PyTorch version: {torch.__version__}\n"
            "Is the topovlm conda env using a CUDA-enabled PyTorch build?\n"
            "Fix: conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia"
        )
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print()

    out_dir  = Path(cfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- FFCV data loaders --------------------------------------------------
    train_beton = cfg["train_beton"]
    val_beton   = cfg["val_beton"]
    for p in (train_beton, val_beton):
        if not Path(p).exists():
            raise FileNotFoundError(
                f".beton file not found: {p}\n"
                "Run write_imagenet_ffcv.py first to create the FFCV dataset files."
            )

    print(f"Loading FFCV data:\n  train: {train_beton}\n  val:   {val_beton}")
    train_loader, val_loader = make_ffcv_loaders(
        train_beton=train_beton,
        val_beton=val_beton,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        device=device,
    )

    # ---- Model --------------------------------------------------------------
    model = make_model(cfg["num_classes"], device)
    opt, sched = make_optimizer(model, cfg)
    start_epoch, best_acc1 = resume_checkpoint(
        model, opt, sched, cfg.get("resume"), device
    )

    # ---- TopoLoss + entropy -------------------------------------------------
    layer_cfg  = cfg["layers"]
    default_lc = cfg["default_layer_cfg"]
    incl_down  = cfg["include_downsample"]

    residual_convs = get_residual_convs(model, include_downsample=incl_down)
    print(f"  Targeting {len(residual_convs)} conv layers: {list(residual_convs.keys())}\n")

    topo_loss_fn = build_topo_loss(model, residual_convs, layer_cfg, default_lc)

    criterion = nn.CrossEntropyLoss().to(device)

    # ---- Train --------------------------------------------------------------
    run_training(
        label=LABEL,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=opt,
        scheduler=sched,
        epochs=cfg["epochs"],
        ckpt_dir=ckpt_dir,
        device=device,
        print_freq=cfg["print_freq"],
        save_freq=cfg["save_freq"],
        topo_loss=topo_loss_fn,
        residual_convs=residual_convs,
        layer_cfg=layer_cfg,
        default_layer_cfg=default_lc,
        start_epoch=start_epoch,
        best_acc1=best_acc1,
    )

    print(f"\nCheckpoints in: {ckpt_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train ResNet-18: TopoLoss + entropy-sparsity (FFCV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(p)
    cli = p.parse_args()

    extra = {
        "output_dir":          cli.output_dir,
        "train_beton":         cli.train_beton,
        "val_beton":           cli.val_beton,
        "num_classes":         cli.num_classes,
        "epochs":              cli.epochs,
        "batch_size":          cli.batch_size,
        "lr":                  cli.lr,
        "momentum":            cli.momentum,
        "weight_decay":        cli.weight_decay,
        "device":              cli.device,
        "num_workers":         cli.num_workers,
        "print_freq":          cli.print_freq,
        "save_freq":           cli.save_freq,
        "resume":              cli.resume,
        "hf_token":            cli.hf_token,
    }
    if cli.include_downsample:
        extra["include_downsample"] = True

    cfg = load_config(cli.config, extra)

    log_path, orig_out, orig_err = setup_logging(LABEL)
    with open(log_path, "w") as fh:
        sys.stdout = _Tee(orig_out, fh)
        sys.stderr = _Tee(orig_err, fh)
        try:
            train(cfg)
        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err

    print(f"Saved run log -> {log_path}")
