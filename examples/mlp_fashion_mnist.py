"""
examples/mlp_fashion_mnist.py

Two-layer MLP with TopoSeedLayer on Fashion-MNIST.

Run:
    python examples/mlp_fashion_mnist.py [--epochs 20] [--batch-size 64]
                                          [--lr 1e-3] [--lambda-cross 0.005]
                                          [--viz-every 5] [--viz-dir outputs/viz_mlp]
                                          [--no-topo] [--device auto]
"""

from __future__ import annotations

import argparse
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
    def __init__(self):
        super().__init__()
        self.layer1 = TopoSeedLayer(
            layer_type="linear",
            in_features=784,
            out_features=256,
            grid_size=4,            # 4×4 = 16 seeds on a ~16×16 sheet
            warmup_steps=200,
            expansion_threshold=0.12,
            death_threshold=0.02,
            death_sustained_steps=150,
            beta=0.7,
            lambda_intra=0.01,
        )
        self.layer2 = TopoSeedLayer(
            layer_type="linear",
            in_features=256,
            out_features=128,
            grid_size=3,            # 3×3 = 9 seeds
            warmup_steps=300,
            expansion_threshold=0.12,
            death_threshold=0.02,
            death_sustained_steps=200,
            beta=0.7,
            lambda_intra=0.01,
        )
        self.classifier = nn.Linear(128, 10)

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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=2, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=pin)

    # ---- Model --------------------------------------------------------------
    if args.no_topo:
        model: nn.Module = BaselineMLP().to(device)
        print("Running BASELINE MLP (no TopoSeed)")
    else:
        model = TopoMLP().to(device)
        print("Running TopoMLP (with TopoSeedLayer)")
        run_correctness_checks(model, device)  # type: ignore[arg-type]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    total_steps = 0
    for epoch in range(1, args.epochs + 1):
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
                ) * args.lambda_cross

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
            print(f"Epoch {epoch:2d}/{args.epochs} | "
                  f"loss={train_loss:.4f} | test_acc={test_acc:.4f} | "
                  f"time={elapsed:.1f}s")
        else:
            topo_model = model  # type: ignore
            print(
                f"Epoch {epoch:2d}/{args.epochs} | "
                f"loss={train_loss:.4f} | test_acc={test_acc:.4f} | "
                f"time={elapsed:.1f}s | "
                f"L1_active={topo_model.layer1.sheet.active_count()} "  # type: ignore
                f"L2_active={topo_model.layer2.sheet.active_count()}"   # type: ignore
            )
            # Periodic visualisations
            if args.viz_every > 0 and epoch % args.viz_every == 0:
                import pathlib
                viz_dir = pathlib.Path(args.viz_dir)
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
        viz_dir = pathlib.Path(args.viz_dir)
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
        # Multi-layer side-by-side selectivity
        plot_multi_layer_selectivity(
            model=topo_model,
            topo_layers=[topo_model.layer1, topo_model.layer2],
            layer_names=["layer1 (256)", "layer2 (128)"],
            out_path=viz_dir / "final_multi_layer_selectivity.png",
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
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-cross", type=float, default=0.005)
    p.add_argument("--viz-every", type=int, default=5,
                   help="Save visualisations every N epochs (0 = only at end)")
    p.add_argument("--viz-dir", type=str, default="outputs/toposeed/viz_mlp",
                   help="Output directory for visualisation PNGs")
    p.add_argument("--no-topo", action="store_true",
                   help="Run baseline MLP without TopoSeed")
    p.add_argument("--device", type=str, default="auto",
                   help="Device: cuda | mps | cpu | auto (default: auto)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
