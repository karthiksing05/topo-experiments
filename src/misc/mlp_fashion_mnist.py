"""
src/mlp_fashion_mnist.py

Two-layer MLP with TopoSeedLayer on Fashion-MNIST.

Run:
    python src/mlp_fashion_mnist.py \
        --config configs/mlp_fashion_mnist.json \
        [--device auto] [--no-topo]

All hyper-parameters live in the JSON config.  See configs/mlp_fashion_mnist.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from toposeed import TopoSeedLayer, cross_layer_coherence, get_device
from toposeed import plot_all, plot_multi_layer_selectivity

FMNIST_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",  "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TopoMLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        l1 = cfg["layer1"]
        l2 = cfg["layer2"]
        self.layer1 = TopoSeedLayer(
            layer_type="linear",
            in_features=l1["in_features"],
            out_features=l1["out_features"],
            grid_size=l1["grid_size"],
            warmup_steps=l1["warmup_steps"],
            expansion_threshold=l1["expansion_threshold"],
            death_threshold=l1["death_threshold"],
            death_sustained_steps=l1["death_sustained_steps"],
            expansion_radius=l1.get("expansion_radius", 1),
            residual_weight=l1.get("residual_weight", 0.5),
            beta=l1["beta"],
            lambda_intra=l1["lambda_intra"],
        )
        self.layer2 = TopoSeedLayer(
            layer_type="linear",
            in_features=l2["in_features"],
            out_features=l2["out_features"],
            grid_size=l2["grid_size"],
            warmup_steps=l2["warmup_steps"],
            expansion_threshold=l2["expansion_threshold"],
            death_threshold=l2["death_threshold"],
            death_sustained_steps=l2["death_sustained_steps"],
            expansion_radius=l2.get("expansion_radius", 1),
            residual_weight=l2.get("residual_weight", 0.5),
            beta=l2["beta"],
            lambda_intra=l2["lambda_intra"],
        )
        self.classifier = nn.Linear(l2["out_features"], 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.classifier(x)


class BaselineMLP(nn.Module):
    """Identical architecture but with standard nn.Linear layers."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def run_correctness_checks(model: TopoMLP, device: torch.device) -> None:
    """Run the five correctness checks described in the spec."""
    print("\n--- Correctness Checks ---")

    # 1. Sheet reshape roundtrip
    TopoSeedLayer.assert_sheet_reshape_roundtrip(16, 16, 784)
    print("  [PASS] Sheet reshape roundtrip")

    # 2. Seed count on init (before any training)
    model.layer1.assert_seed_count()
    model.layer2.assert_seed_count()
    print("  [PASS] Seed count == grid_size^2")

    # 3. Dormant neurons contribute zero
    x = torch.randn(4, 784, device=device)
    model.layer1.assert_dormant_zero(x)
    print("  [PASS] Dormant neurons contribute zero")

    print("--------------------------\n")


def train(args: argparse.Namespace) -> None:
    # ---- Load config --------------------------------------------------------
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.config,
    ) if not os.path.isabs(args.config) else args.config

    with open(cfg_path) as f:
        cfg = json.load(f)

    tcfg = cfg["training"]
    vcfg = cfg["viz"]

    epochs      = tcfg["epochs"]
    batch_size  = tcfg["batch_size"]
    lr          = tcfg["lr"]
    lambda_cross = tcfg["lambda_cross"]
    viz_every   = vcfg["viz_every"]
    viz_dir_str = vcfg["viz_dir"]

    print(f"Config: {cfg_path}")

    device = get_device(args.device)
    print(f"Device: {device}")

    # ---- Data ---------------------------------------------------------------
    transform = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
    train_ds = torchvision.datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transform
    )
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, num_workers=2, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=pin)

    # ---- Model --------------------------------------------------------------
    if args.no_topo:
        model: nn.Module = BaselineMLP().to(device)
        print("Running BASELINE MLP (no TopoSeed)")
    else:
        model = TopoMLP(cfg).to(device)
        print("Running TopoMLP (with TopoSeedLayer)")
        run_correctness_checks(model, device)  # type: ignore[arg-type]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_steps = 0
    for epoch in range(1, epochs + 1):
        model.train()
        if not args.no_topo:
            model.layer1.reset_epoch_stats()   # type: ignore[attr-defined]
            model.layer2.reset_epoch_stats()   # type: ignore[attr-defined]

        running_loss = 0.0
        t0 = time.time()

        for step, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(imgs)
            task_loss = criterion(logits, labels)

            if not args.no_topo:
                topo_model: TopoMLP = model  # type: ignore
                # Compute cross-layer coherence regulariser
                cross_loss = cross_layer_coherence(
                    topo_model.layer1.sheet,
                    topo_model.layer2.sheet,
                    topo_model.layer2.sheet.get_active_weight_matrix(),
                ) * lambda_cross

                total_loss = (
                    task_loss
                    + topo_model.layer1.get_reg_loss()
                    + topo_model.layer2.get_reg_loss()
                    + cross_loss
                )
            else:
                total_loss = task_loss

            total_loss.backward()
            optimizer.step()

            # TopoSeed buffer update (after backward, before next forward)
            if not args.no_topo:
                topo_model.layer1.update_buffers()
                topo_model.layer2.update_buffers()

            running_loss += task_loss.item()
            total_steps += 1

            # No expansion during warmup check (first epoch only)
            if epoch == 1 and not args.no_topo:
                topo_model.layer1.assert_no_expansion_during_warmup()
                topo_model.layer2.assert_no_expansion_during_warmup()

            if total_steps % 100 == 0 and not args.no_topo:
                print(f"  Step {total_steps} | layer1: {topo_model.layer1.get_stats()}")
                print(f"  Step {total_steps} | layer2: {topo_model.layer2.get_stats()}")

        elapsed = time.time() - t0
        train_loss = running_loss / len(train_loader)
        test_acc = evaluate(model, test_loader, device)

        if args.no_topo:
            print(f"Epoch {epoch:2d}/{epochs} | "
                  f"loss={train_loss:.4f} | test_acc={test_acc:.4f} | "
                  f"time={elapsed:.1f}s")
        else:
            topo_model = model  # type: ignore
            print(
                f"Epoch {epoch:2d}/{epochs} | "
                f"loss={train_loss:.4f} | test_acc={test_acc:.4f} | "
                f"time={elapsed:.1f}s | "
                f"L1_active={topo_model.layer1.sheet.active_count()} "  # type: ignore
                f"L2_active={topo_model.layer2.sheet.active_count()}"   # type: ignore
            )
            # Periodic visualisations → debug/epoch{N:03d}/ subdirs
            if viz_every > 0 and epoch % viz_every == 0:
                import pathlib
                viz_dir = pathlib.Path(viz_dir_str)
                for lname, layer in [("layer1", topo_model.layer1),
                                      ("layer2", topo_model.layer2)]:
                    plot_all(
                        model=topo_model,
                        layer=layer,
                        layer_name=lname,
                        dataloader=test_loader,
                        class_names=FMNIST_CLASSES,
                        device=device,
                        out_dir=viz_dir,
                        epoch=epoch,
                    )

    print(f"\nFinal test accuracy: {evaluate(model, test_loader, device):.4f}")

    # ---- Visualisations -----------------------------------------------------
    if not args.no_topo:
        topo_model = model  # type: ignore
        import pathlib
        viz_dir = pathlib.Path(viz_dir_str)
        print(f"\nSaving final visualisations to {viz_dir} ...")
        # Per-layer weight sheets, masks, and selectivity maps
        for lname, layer in [("layer1", topo_model.layer1),
                              ("layer2", topo_model.layer2)]:
            plot_all(
                model=topo_model,
                layer=layer,
                layer_name=lname,
                dataloader=test_loader,
                class_names=FMNIST_CLASSES,
                device=device,
                out_dir=viz_dir,
                epoch=None,
                prefix="final_",
            )
        # Multi-layer side-by-side selectivity → selectivities subdir
        plot_multi_layer_selectivity(
            model=topo_model,
            topo_layers=[topo_model.layer1, topo_model.layer2],
            layer_names=["layer1 (256)", "layer2 (128)"],
            out_path=viz_dir / "selectivities" / "final_multi_layer_selectivity.png",
            dataloader=test_loader,
            class_names=FMNIST_CLASSES,
            device=device,
            title="Fashion-MNIST — Multi-Layer Selectivity",
        )
        print("  Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TopoSeed MLP on Fashion-MNIST")
    p.add_argument(
        "--config", type=str,
        default="configs/mlp_fashion_mnist.json",
        help="Path to JSON config file (relative to repo root or absolute)",
    )
    p.add_argument("--no-topo", action="store_true",
                   help="Run baseline MLP without TopoSeed")
    p.add_argument("--device", type=str, default="auto",
                   help="Device: cuda | mps | cpu | auto (default: auto)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
