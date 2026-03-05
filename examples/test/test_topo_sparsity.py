"""Evaluation script for train_topo_sparsity experiments.

Loads the three trained model checkpoints (topo, topo_only, baseline) and
generates all visualisation figures under outputs/train_topo_sparsity/:

    selectivity/   – per-model selectivity maps, comparison figures, subset comparisons
    activations/   – activation cortical sheets, per-category activation comparisons
    similarity/    – pairwise SSIM comparison matrices
    debug/         – debug cortical-sheet figures

Usage
-----
python examples/test/test_topo_sparsity.py
python examples/test/test_topo_sparsity.py --config configs/train_topo_sparsity.json
python examples/test/test_topo_sparsity.py \\
    --checkpoint-topo      outputs/train_topo_sparsity/checkpoints/best_topo.pt \\
    --checkpoint-topo-only outputs/train_topo_sparsity/checkpoints/best_topo_only.pt \\
    --checkpoint-base      outputs/train_topo_sparsity/checkpoints/best_baseline.pt

All paths default to the project root.  Override any path via the corresponding
CLI flag (run with --help for the full list).
"""

# -- Imports -------------------------------------------------------------------

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

# Make train/ importable so we can reuse the shared model & vis functions.
_TRAIN_DIR = Path(__file__).resolve().parents[1] / "train"
sys.path.insert(0, str(_TRAIN_DIR))

import matplotlib
matplotlib.use("Agg")

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from train_topo_sparsity import (           # shared definitions
    BASE_DIR,
    SimpleNN,
    FMNIST_CLASSES,
    VIS_LAYER_NAMES,
    TOPO_LAYER_NAMES,
    save_debug_cortical_sheets,
    save_selectivity_maps,
    save_activation_cortical_sheets,
    save_per_category_activation_comparisons,
    save_ssim_comparison_matrices,
    save_comparison_figure,
    save_subset_comparison_figure,
)

# -- Defaults ------------------------------------------------------------------

OUTPUT_DIR = BASE_DIR / "outputs" / "train_topo_sparsity"

_DEFAULT_CONFIG = {
    "data_dir":             None,       # null → BASE_DIR/data
    "output_dir":           None,       # null → OUTPUT_DIR
    "hidden_size":          256,
    "batch_size":           256,
    "device":               "cuda:0",
    "layers": {
        "fc1": {
            "topo_scale": 10.0, "factor_h": 4.0, "factor_w": 4.0,
            "lambda_kl":   1.0, "lambda_entropy": 1.0, "temperature": 3.0,
        },
    },
    # checkpoint overrides (None → auto-detected from ckpt_dir)
    "checkpoint_topo":      None,
    "checkpoint_topo_only": None,
    "checkpoint_base":      None,
}


# -- Config loading ------------------------------------------------------------

def get_config() -> dict:
    """Parse CLI arguments, overlay JSON config, return resolved config dict."""
    p = argparse.ArgumentParser(
        description="Evaluate / visualise trained train_topo_sparsity checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_cfg = str(BASE_DIR / "configs" / "train_topo_sparsity.json")
    p.add_argument("--config",                type=str, default=default_cfg,
                   help="Path to JSON config file (default: configs/train_topo_sparsity.json)")
    p.add_argument("--data-dir",              type=str, default=None)
    p.add_argument("--output-dir",            type=str, default=None,
                   help="Root output directory (default from config or project root)")
    p.add_argument("--hidden-size",           type=int, default=None)
    p.add_argument("--batch-size",            type=int, default=None)
    p.add_argument("--device",                type=str, default=None)
    p.add_argument("--checkpoint-topo",       type=str, default=None,
                   help="Path to topo checkpoint (default: <output_dir>/checkpoints/best_topo.pt)")
    p.add_argument("--checkpoint-topo-only",  type=str, default=None,
                   help="Path to topo-only checkpoint")
    p.add_argument("--checkpoint-base",       type=str, default=None,
                   help="Path to baseline checkpoint")
    cli = p.parse_args()

    cfg = copy.deepcopy(_DEFAULT_CONFIG)

    cfg_path = Path(cli.config)
    if cfg_path.exists():
        with open(cfg_path) as fh:
            file_cfg = json.load(fh)
        for key, val in file_cfg.items():
            if key == "layers" and isinstance(val, dict):
                for lname, lvals in val.items():
                    if lname in cfg["layers"]:
                        cfg["layers"][lname].update(lvals)
                    else:
                        cfg["layers"][lname] = lvals
            elif key not in ("_comment",):
                cfg[key] = val
        print(f"Config loaded from: {cfg_path}")
    else:
        print(f"Config file not found ({cfg_path}), using built-in defaults.")

    cli_map = {
        "data_dir":             cli.data_dir,
        "output_dir":           cli.output_dir,
        "hidden_size":          cli.hidden_size,
        "batch_size":           cli.batch_size,
        "device":               cli.device,
        "checkpoint_topo":      cli.checkpoint_topo,
        "checkpoint_topo_only": cli.checkpoint_topo_only,
        "checkpoint_base":      cli.checkpoint_base,
    }
    for key, val in cli_map.items():
        if val is not None:
            cfg[key] = val

    if cfg["data_dir"]   is None:
        cfg["data_dir"]   = str(BASE_DIR / "data")
    if cfg["output_dir"] is None:
        cfg["output_dir"] = str(OUTPUT_DIR)

    return cfg


# -- Checkpoint loading --------------------------------------------------------

def _load_model(ckpt_path: Path, hidden_size: int, device: str) -> SimpleNN:
    """Restore a SimpleNN from a checkpoint; print best val acc."""
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SimpleNN(hidden_size=hidden_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    best_acc = ckpt.get("best_acc", float("nan"))
    print(f"  Loaded  {ckpt_path}  (best val acc: {best_acc:.2f}%)")
    return model


# -- Main evaluation entry point -----------------------------------------------

def evaluate(cfg: dict) -> None:
    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = cfg["device"]
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    out_dir  = Path(cfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints"

    # ── Val data loader ───────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    val_ds = datasets.FashionMNIST(
        cfg["data_dir"], train=False, download=True, transform=transform
    )
    _pin = device.startswith("cuda")
    val_loader = DataLoader(
        val_ds, batch_size=cfg.get("batch_size", 256),
        shuffle=False, num_workers=0, pin_memory=_pin,
    )
    print(f"Val dataset: {len(val_ds):,} samples  |  {len(FMNIST_CLASSES)} classes\n")

    hidden_size = cfg.get("hidden_size", 256)
    layer_cfg   = cfg["layers"]

    # ── Resolve checkpoint paths ──────────────────────────────────────────────
    def _resolve_ckpt(cfg_key: str, default_name: str) -> Path:
        override = cfg.get(cfg_key)
        if override:
            path = Path(override)
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            return path
        candidate = ckpt_dir / default_name
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Checkpoint not found: {candidate}.\n"
            f"Train the model first (examples/train/train_topo_sparsity.py) or "
            f"supply --{cfg_key.replace('_', '-')}."
        )

    topo_ckpt      = _resolve_ckpt("checkpoint_topo",      "best_topo.pt")
    topo_only_ckpt = _resolve_ckpt("checkpoint_topo_only", "best_topo_only.pt")
    base_ckpt      = _resolve_ckpt("checkpoint_base",      "best_baseline.pt")

    print("=" * 65)
    print("  Loading model checkpoints ...")
    print("=" * 65)
    topo_model      = _load_model(topo_ckpt,      hidden_size, device)
    topo_only_model = _load_model(topo_only_ckpt, hidden_size, device)
    base_model      = _load_model(base_ckpt,      hidden_size, device)

    all_models  = {
        "topo":      topo_model,
        "topo-only": topo_only_model,
        "baseline":  base_model,
    }
    topo_models = {"topo": topo_model, "topo-only": topo_only_model}

    # ── Output subdirectories ─────────────────────────────────────────────────
    selectivity_dir = out_dir / "selectivity"
    activations_dir = out_dir / "activations"
    similarity_dir  = out_dir / "similarity"
    debug_dir       = out_dir / "debug"
    for d in (selectivity_dir, activations_dir, similarity_dir, debug_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Selectivity maps ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Generating selectivity maps ...")
    print("=" * 65)
    for lbl, mdl in all_models.items():
        save_selectivity_maps(mdl, val_loader, selectivity_dir, device, tag=lbl)
    save_comparison_figure(all_models, val_loader, selectivity_dir, device)

    # Subset comparisons: topo vs topo-only only
    # Tops: T-shirt(0), Pullover(2), Coat(4), Shirt(6)
    save_subset_comparison_figure(
        topo_models, val_loader, selectivity_dir, device,
        class_indices=[0, 2, 4, 6],
        tag="tops",
        subtitle="Tops (T-shirt, Pullover, Coat, Shirt)",
    )
    # Shoes + Bag: Sandal(5), Sneaker(7), AnkleBoot(9), Bag(8)
    save_subset_comparison_figure(
        topo_models, val_loader, selectivity_dir, device,
        class_indices=[5, 7, 9, 8],
        tag="shoes_bag",
        subtitle="Shoes & Bag (Sandal, Sneaker, AnkleBoot, Bag)",
    )
    # Lower-body / silhouette: Trouser(1), Dress(3)
    save_subset_comparison_figure(
        topo_models, val_loader, selectivity_dir, device,
        class_indices=[1, 3],
        tag="trouser_dress",
        subtitle="Trouser & Dress",
    )

    # ── Similarity (SSIM) matrices ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Generating SSIM comparison matrices ...")
    print("=" * 65)
    save_ssim_comparison_matrices(all_models, val_loader, similarity_dir, device)

    # ── Activation cortical sheets ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Generating activation cortical-sheet figures ...")
    print("=" * 65)
    save_activation_cortical_sheets(all_models, val_loader, activations_dir, device)
    save_per_category_activation_comparisons(topo_models, val_loader, activations_dir, device)

    # ── Debug cortical sheets ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Generating debug cortical-sheet figures ...")
    print("=" * 65)
    save_debug_cortical_sheets(
        all_models, val_loader,
        layer_cfg=layer_cfg,
        out_dir=debug_dir, device=device,
    )

    print(f"\nAll outputs in: {out_dir}")
    print(f"  selectivity/ -> {selectivity_dir}")
    print(f"  activations/ -> {activations_dir}")
    print(f"  similarity/  -> {similarity_dir}")
    print(f"  debug/       -> {debug_dir}")


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    cfg = get_config()

    # Tee stdout/stderr to a timestamped log file
    log_dir = BASE_DIR / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"test_topo_sparsity-{ts}.txt"

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams:
                try:
                    s.write(data)
                except Exception:
                    pass
        def flush(self):
            for s in self._streams:
                try:
                    s.flush()
                except Exception:
                    pass
        def isatty(self):
            for s in self._streams:
                if hasattr(s, "isatty") and s.isatty():
                    return True
            return False

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    with open(log_path, "w") as fh:
        sys.stdout = _Tee(orig_stdout, fh)
        sys.stderr = _Tee(orig_stderr, fh)
        try:
            evaluate(cfg)
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

    print(f"Saved run log -> {log_path}")
